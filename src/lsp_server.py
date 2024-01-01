from functools import partial
import logging
import os
import socketserver
import threading
import uuid
from typing import List, Dict, Any
import ujson as json

from pylsp_jsonrpc.dispatchers import MethodDispatcher
from pylsp_jsonrpc.endpoint import Endpoint
from pylsp_jsonrpc.streams import JsonRpcStreamReader, JsonRpcStreamWriter

from pylsp import lsp, _utils, uris
from pylsp.config import config
from pylsp.workspace import Workspace, Document, Notebook, Cell

log = logging.getLogger(__name__)


LINT_DEBOUNCE_S = 1.5  # 500 ms
PARENT_PROCESS_WATCH_INTERVAL = 10  # 10 s
MAX_WORKERS = 64
PYTHON_FILE_EXTENSIONS = (".py", ".pyi")
CONFIG_FILEs = ("pycodestyle.cfg", "setup.cfg", "tox.ini", ".flake8")


class _StreamHandlerWrapper(socketserver.StreamRequestHandler):
    """A wrapper class that is used to construct a custom handler class."""

    delegate = None

    def setup(self):
        super().setup()
        self.delegate = self.DELEGATE_CLASS(self.rfile, self.wfile)

    def handle(self):
        try:
            self.delegate.start()
        except OSError as e:
            if os.name == "nt":
                # Catch and pass on ConnectionResetError when parent process
                # dies
                # pylint: disable=no-member, undefined-variable
                if isinstance(e, WindowsError) and e.winerror == 10054:
                    pass

        self.SHUTDOWN_CALL()


def start_tcp_lang_server(bind_addr, port, check_parent_process, handler_class):
    if not issubclass(handler_class, PythonLSPServer):
        raise ValueError("Handler class must be an instance of PythonLSPServer")

    def shutdown_server(check_parent_process, *args):
        # pylint: disable=unused-argument
        if check_parent_process:
            log.debug("Shutting down server")
            # Shutdown call must be done on a thread, to prevent deadlocks
            stop_thread = threading.Thread(target=server.shutdown)
            stop_thread.start()

    # Construct a custom wrapper class around the user's handler_class
    wrapper_class = type(
        handler_class.__name__ + "Handler",
        (_StreamHandlerWrapper,),
        {
            "DELEGATE_CLASS": partial(
                handler_class, check_parent_process=check_parent_process
            ),
            "SHUTDOWN_CALL": partial(shutdown_server, check_parent_process),
        },
    )

    server = socketserver.TCPServer(
        (bind_addr, port), wrapper_class, bind_and_activate=False
    )
    server.allow_reuse_address = True

    try:
        server.server_bind()
        server.server_activate()
        log.info("Serving %s on (%s, %s)", handler_class.__name__, bind_addr, port)
        server.serve_forever()
    finally:
        log.info("Shutting down")
        server.server_close()


def start_io_lang_server(rfile, wfile, check_parent_process, handler_class):
    if not issubclass(handler_class, PythonLSPServer):
        raise ValueError("Handler class must be an instance of PythonLSPServer")
    log.info("Starting %s IO language server", handler_class.__name__)
    server = handler_class(rfile, wfile, check_parent_process)
    server.start()


def start_ws_lang_server(port, check_parent_process, handler_class):
    if not issubclass(handler_class, PythonLSPServer):
        raise ValueError("Handler class must be an instance of PythonLSPServer")

    # pylint: disable=import-outside-toplevel

    # imports needed only for websockets based server
    try:
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        import websockets
    except ImportError as e:
        raise ImportError(
            "websocket modules missing. Please run pip install 'python-lsp-server[websockets]"
        ) from e

    with ThreadPoolExecutor(max_workers=10) as tpool:

        async def pylsp_ws(websocket):
            log.debug("Creating LSP object")

            # creating a partial function and suppling the websocket connection
            response_handler = partial(send_message, websocket=websocket)

            # Not using default stream reader and writer.
            # Instead using a consumer based approach to handle processed requests
            pylsp_handler = handler_class(
                rx=None,
                tx=None,
                consumer=response_handler,
                check_parent_process=check_parent_process,
            )

            async for message in websocket:
                try:
                    log.debug("consuming payload and feeding it to LSP handler")
                    # pylint: disable=c-extension-no-member
                    request = json.loads(message)
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(tpool, pylsp_handler.consume, request)
                except Exception as e:  # pylint: disable=broad-except
                    log.exception("Failed to process request %s, %s", message, str(e))

        def send_message(message, websocket):
            """Handler to send responses of  processed requests to respective web socket clients"""
            try:
                # pylint: disable=c-extension-no-member
                payload = json.dumps(message, ensure_ascii=False)
                asyncio.run(websocket.send(payload))
            except Exception as e:  # pylint: disable=broad-except
                log.exception("Failed to write message %s, %s", message, str(e))

        async def run_server():
            async with websockets.serve(pylsp_ws, port=port):
                # runs forever
                await asyncio.Future()

        asyncio.run(run_server())


class PythonLSPServer(MethodDispatcher):
    """Implementation of the Microsoft VSCode Language Server Protocol
    https://github.com/Microsoft/language-server-protocol/blob/master/versions/protocol-1-x.md
    """

    # pylint: disable=too-many-public-methods,redefined-builtin

    def __init__(
        self, rx, tx, check_parent_process=False, consumer=None, *, endpoint_cls=None
    ):
        self.workspace = None
        self.config = None
        self.root_uri = None
        self.watching_thread = None
        self.workspaces = {}
        self.uri_workspace_mapper = {}

        self._check_parent_process = check_parent_process

        if rx is not None:
            self._jsonrpc_stream_reader = JsonRpcStreamReader(rx)
        else:
            self._jsonrpc_stream_reader = None

        if tx is not None:
            self._jsonrpc_stream_writer = JsonRpcStreamWriter(tx)
        else:
            self._jsonrpc_stream_writer = None

        endpoint_cls = endpoint_cls or Endpoint

        # if consumer is None, it is assumed that the default streams-based approach is being used
        if consumer is None:
            self._endpoint = endpoint_cls(
                self, self._jsonrpc_stream_writer.write, max_workers=MAX_WORKERS
            )
        else:
            self._endpoint = endpoint_cls(self, consumer, max_workers=MAX_WORKERS)

        self._dispatchers = []
        self._shutdown = False

    def start(self):
        """Entry point for the server."""
        self._jsonrpc_stream_reader.listen(self._endpoint.consume)

    def consume(self, message):
        """Entry point for consumer based server. Alternative to stream listeners."""
        # assuming message will be JSON
        self._endpoint.consume(message)

    def __getitem__(self, item):
        """Override getitem to fallback through multiple dispatchers."""
        if self._shutdown and item != "exit":
            # exit is the only allowed method during shutdown
            log.debug("Ignoring non-exit method during shutdown: %s", item)
            item = "invalid_request_after_shutdown"

        try:
            return super().__getitem__(item)
        except KeyError:
            # Fallback through extra dispatchers
            for dispatcher in self._dispatchers:
                try:
                    return dispatcher[item]
                except KeyError:
                    continue

        raise KeyError()

    def m_shutdown(self, **_kwargs):
        for workspace in self.workspaces.values():
            workspace.close()
        self._shutdown = True

    def m_invalid_request_after_shutdown(self, **_kwargs):
        return {
            "error": {
                "code": lsp.ErrorCodes.InvalidRequest,
                "message": "Requests after shutdown are not valid",
            }
        }

    def m_exit(self, **_kwargs):
        self._endpoint.shutdown()
        if self._jsonrpc_stream_reader is not None:
            self._jsonrpc_stream_reader.close()
        if self._jsonrpc_stream_writer is not None:
            self._jsonrpc_stream_writer.close()

    def _match_uri_to_workspace(self, uri):
        workspace_uri = _utils.match_uri_to_workspace(uri, self.workspaces)
        return self.workspaces.get(workspace_uri, self.workspace)

    def _hook(self, hook_name, doc_uri=None, **kwargs):
        """Calls hook_name and returns a list of results from all registered handlers"""
        workspace = self._match_uri_to_workspace(doc_uri)
        doc = workspace.get_document(doc_uri) if doc_uri else None
        hook_handlers = self.config.plugin_manager.subset_hook_caller(
            hook_name, self.config.disabled_plugins
        )
        return hook_handlers(
            config=self.config, workspace=workspace, document=doc, **kwargs
        )

    def capabilities(self):
        server_capabilities = {
            "codeActionProvider": False,
            "codeLensProvider": {
                "resolveProvider": False,  # We may need to make this configurable
            },
            "completionProvider": {
                "resolveProvider": True,  # We could know everything ahead of time, but this takes time to transfer
                "triggerCharacters": [".", " "],
            },
            "documentFormattingProvider": False,
            "documentHighlightProvider": False,
            "documentRangeFormattingProvider": False,
            "documentSymbolProvider": False,
            "definitionProvider": False,
            "executeCommandProvider": {
                # "commands": flatten(self._hook("pylsp_commands"))
            },
            "hoverProvider": True,
            "referencesProvider": False,
            "renameProvider": False,
            "foldingRangeProvider": False,
            # "signatureHelpProvider": {"triggerCharacters": ["(", ",", "="]},
            "textDocumentSync": {
                "change": lsp.TextDocumentSyncKind.INCREMENTAL,
                "save": {
                    "includeText": True,
                },
                "openClose": True,
            },
            # "notebookDocumentSync": {
            #     "notebookSelector": [{"cells": [{"language": "python"}]}]
            # },
            "workspace": {
                "workspaceFolders": {"supported": True, "changeNotifications": True}
            },
            # "experimental": merge(self._hook("pylsp_experimental_capabilities")),
        }
        log.info("Server capabilities: %s", server_capabilities)
        return server_capabilities

    def m_initialize(
        self,
        processId=None,
        rootUri=None,
        rootPath=None,
        initializationOptions=None,
        workspaceFolders=None,
        **_kwargs,
    ):
        log.debug(
            "Language server initialized with %s %s %s %s",
            processId,
            rootUri,
            rootPath,
            initializationOptions,
        )
        if rootUri is None:
            rootUri = uris.from_fs_path(rootPath) if rootPath is not None else ""

        self.workspaces.pop(self.root_uri, None)
        self.root_uri = rootUri
        self.config = config.Config(
            rootUri,
            initializationOptions or {},
            processId,
            _kwargs.get("capabilities", {}),
        )
        self.workspace = Workspace(rootUri, self._endpoint, self.config)
        self.workspaces[rootUri] = self.workspace
        if workspaceFolders:
            for folder in workspaceFolders:
                uri = folder["uri"]
                if uri == rootUri:
                    # Already created
                    continue
                workspace_config = config.Config(
                    uri,
                    self.config._init_opts,
                    self.config._process_id,
                    self.config._capabilities,
                )
                workspace_config.update(self.config._settings)
                self.workspaces[uri] = Workspace(uri, self._endpoint, workspace_config)

        self._dispatchers = self._hook("pylsp_dispatchers")
        self._hook("pylsp_initialize")

        if (
            self._check_parent_process
            and processId is not None
            and self.watching_thread is None
        ):

            def watch_parent_process(pid):
                # exit when the given pid is not alive
                if not _utils.is_process_alive(pid):
                    log.info("parent process %s is not alive, exiting!", pid)
                    self.m_exit()
                else:
                    threading.Timer(
                        PARENT_PROCESS_WATCH_INTERVAL, watch_parent_process, args=[pid]
                    ).start()

            self.watching_thread = threading.Thread(
                target=watch_parent_process, args=(processId,)
            )
            self.watching_thread.daemon = True
            self.watching_thread.start()
        # Get our capabilities
        return {
            "capabilities": self.capabilities(),
            "serverInfo": {
                "name": "pylsp",
                "version": "0.0.1",
            },
        }

    def m_initialized(self, **_kwargs):
        self._hook("pylsp_initialized")


    def completions(self, doc_uri, position):
        workspace = self._match_uri_to_workspace(doc_uri)
        document = workspace.get_document(doc_uri)
        ignored_names = None
        if isinstance(document, Cell):
            # We need to get the ignored names from the whole notebook document
            notebook_document = workspace.get_maybe_document(document.notebook_uri)
            ignored_names = notebook_document.jedi_names(doc_uri)
        completions = self._hook(
            "pylsp_completions", doc_uri, position=position, ignored_names=ignored_names
        )
        return {"isIncomplete": False, "items": flatten(completions)}

    def completion_item_resolve(self, completion_item):
        doc_uri = completion_item.get("data", {}).get("doc_uri", None)
        return self._hook(
            "pylsp_completion_item_resolve", doc_uri, completion_item=completion_item
        )


    def m_completion_item__resolve(self, **completionItem):
        return self.completion_item_resolve(completionItem)


    def m_text_document__did_close(self, textDocument=None, **_kwargs):
        workspace = self._match_uri_to_workspace(textDocument["uri"])
        workspace.publish_diagnostics(textDocument["uri"], [])
        workspace.rm_document(textDocument["uri"])

    def m_text_document__did_open(self, textDocument=None, **_kwargs):
        workspace = self._match_uri_to_workspace(textDocument["uri"])
        workspace.put_document(
            textDocument["uri"],
            textDocument["text"],
            version=textDocument.get("version"),
        )
        self._hook("pylsp_document_did_open", textDocument["uri"])
        self.lint(textDocument["uri"], is_saved=True)

    def m_text_document__did_change(
        self, contentChanges=None, textDocument=None, **_kwargs
    ):
        workspace = self._match_uri_to_workspace(textDocument["uri"])
        for change in contentChanges:
            workspace.update_document(
                textDocument["uri"], change, version=textDocument.get("version")
            )
        self.lint(textDocument["uri"], is_saved=False)

def flatten(list_of_lists):
    return [item for lst in list_of_lists for item in lst]


def merge(list_of_dicts):
    return {k: v for dictionary in list_of_dicts for k, v in dictionary.items()}
