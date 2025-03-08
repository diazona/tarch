#!/usr/bin/env python

import argparse
import itertools
import pathlib
import sys
from dataclasses import dataclass, field
from functools import cached_property
from rich.style import Style
from rich.text import Text
from textual import log, work
from textual.app import App, ComposeResult
from textual.containers import Center, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen, Screen
from textual.strip import Strip
from textual.widgets import DirectoryTree, Footer, Header, Input, Label, ProgressBar, Static, Tree
from textual.widgets.tree import NodeID, TreeDataType, TreeNode
from textual.worker import Worker, WorkerState
from typing import Iterable, Iterator, TypeAlias, Union


@dataclass(frozen=True)
class VirtualDirEntry:
    name: str
    parent: "VirtualDirEntry | None"
    children: dict[str, "VirtualDirEntry"] | None = field(default_factory=dict, compare=False)

    def add_child(self, name: str, is_dir: bool = False) -> "VirtualDirEntry":
        if "/" in name:
            raise ValueError("'/' not allowed in filename")
        elif self.children is None:
            raise ValueError("Cannot add child to file")
        child: VirtualDirEntry
        try:
            child = self.children[name]
        except KeyError:
            child = self.children[name] = (VirtualDirEntry(name, self, None) if not is_dir else VirtualDirEntry(name, self))
        return child

    def is_dir(self) -> bool:
        return self.children is not None

    @cached_property
    def descendant_count(self) -> int:
        if self.children is None:
            return 0
        return len(self.children) + sum(c.descendant_count for c in self.children.values())

    @cached_property
    def path(self) -> str:
        path = ""
        if self.parent:
            path += self.parent.path
        path += self.name
        if self.is_dir():
            path += "/"
        return path

    def __getitem__(self, name: str) -> "VirtualDirEntry":
        if self.children is None:
            raise TypeError("Not a directory")
        return self.children[name]

    def __delitem__(self, name: str) -> None:
        if not self.children:
            raise TypeError("Not a directory")
        # raise KeyError if it doesn't exist
        del self.children[name]

    def __repr__(self) -> str:
        if self.children is None:
            children_repr = repr(None)
        else:
            children_repr = f"{{({len(self.children)} entries)}}"
        if self.parent is None:
            parent_repr = repr(None)
        else:
            parent_repr = f"{self.parent.__class__.__name__}(name={self.parent.name}, ...)"
        return f"{self.__class__.__name__}(name={self.name!r}, parent={parent_repr}, children={children_repr})"


class MarkableTree(Tree[TreeDataType]):
    BINDINGS = [
        ("m", "toggle_mark"),
    ]

    COMPONENT_CLASSES: set[str] = {  # type: ignore[misc]
        "markable-tree--label-marked",
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.marked: set[TreeDataType] = set()

    def action_toggle_mark(self) -> None:
        node = self.cursor_node
        if not node or not node.data:
            return
        elif node.data in self.marked:
            self.marked.remove(node.data)
        else:
            self.marked.add(node.data)

    def render_label(self, node: TreeNode[TreeDataType], base_style: Style, style: Style) -> Text:
        if node.data in self.marked:
            style = style + self.get_component_rich_style("markable-tree--label-marked", partial=True)
        return super().render_label(node, base_style, style)


class VirtualFileTree(MarkableTree[VirtualDirEntry]):
    def __init__(self, root: VirtualDirEntry, **kwargs):
        super().__init__(label=root.name, data=root, **kwargs)

    def on_tree_node_expanded(self, event: Tree.NodeExpanded) -> None:
        dir_entry = event.node.data
        if not dir_entry or not dir_entry.is_dir():
            return
        sorted_children = sorted(dir_entry.children.values(), key=lambda c: (not c.is_dir(), c.name.lower()))
        for child in sorted_children:
            is_dir = child.is_dir()
            if is_dir:
                label = f"{child.name} ({child.descendant_count})"
            else:
                label = child.name
            event.node.add(label=label, data=child, expand=False, allow_expand=is_dir)

    def on_tree_node_collapsed(self, event: Tree.NodeCollapsed) -> None:
        event.node.remove_children()


class LoadingScreen(Screen[VirtualDirEntry | None]):
    def __init__(self, paths: Iterable[str], limit: int | None = None):
        super().__init__()
        self._paths: Iterable[str] = paths
        self._limit: int | None = limit

    def on_mount(self) -> None:
        self._build_model()

    def compose(self) -> ComposeResult:
        with Center():
            yield Label("Loading...")
            yield ProgressBar(total=self._limit)

    @work(thread=True)
    def _build_model(self) -> VirtualDirEntry:
        root = VirtualDirEntry("<root>", None)
        progress_bar = self.query_one(ProgressBar)
        try:
            update_interval = len(self._paths) // 200  # type: ignore[arg-type]
            self.log(update_interval=update_interval, source="dynamic")
        except TypeError:
            update_interval = 100
            self.log(update_interval=update_interval, source="fixed")
        for i, path in enumerate(self._paths, start=1):
            components = path.split("/")
            current = root
            for component in components[:-1]:
                child = current.add_child(component, is_dir=True)
                current = child
            if components[-1]:
                # components[-1] is only nonempty if it's a file, not a directory
                current.add_child(components[-1], is_dir=False)
            if i % update_interval == 0:
                progress_bar.update(progress=i)
        return root

    def on_worker_state_changed(self, event: Worker.StateChanged):
        if event.state in (WorkerState.ERROR, WorkerState.CANCELLED):
            self.dismiss(None)
        elif event.state == WorkerState.SUCCESS:
            self.dismiss(event.worker.result)


class OnlyDirectoryTree(DirectoryTree):
    BINDINGS = [("h", "toggle_hidden")]

    show_hidden_files: reactive[bool] = reactive(False)

    def on_mount(self) -> None:
        try:
            rel_cwd = pathlib.Path.cwd().relative_to(self.path)
        except ValueError:
            self.log("Unable to compute relative path", cwd=pathlib.Path.cwd(), root=self.path)
            return
        current_node = self.root
        try:
            for component in rel_cwd.parts:
                if not current_node:
                    return
                child_node = next(node for node in current_node.children if component == node.label)
                child_node.expand()
                current_node = child_node
        except StopIteration:
            self.log("Finished iteration", locals=locals())


    def filter_paths(self, paths: Iterable[pathlib.Path]) -> Iterable[pathlib.Path]:
        if self.show_hidden_files:
            return (p for p in paths if p.is_dir())
        else:
            return (p for p in paths if p.is_dir() and not p.name.startswith("."))

    def action_toggle_hidden(self) -> None:
        self.show_hidden_files = not self.show_hidden_files


class FileChooserScreen(ModalScreen[pathlib.Path]):
    BINDINGS = [("escape", "dismiss")]

    def compose(self) -> ComposeResult:
        with Center():
            yield Label("Destination")
            yield OnlyDirectoryTree(pathlib.Path("/home/diazona"))
            yield Input()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.action_submit()

    def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        self.focus_next()

    def action_submit(self) -> None:
        tree = self.query_one(DirectoryTree)
        try:
            directory = tree.cursor_node.data.path  # type: ignore[union-attr]
        except AttributeError:
            return
        field = self.query_one(Input)
        name = field.value
        self.dismiss(directory / name)


class PathTreeScreen(Screen):
    BINDINGS = [("w", "write_marked")]

    def __init__(self, root: VirtualDirEntry):
        super().__init__()
        self._root = root

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Header()
            vft = VirtualFileTree(self._root)
            vft.show_root = False
            vft.root.expand()
            yield vft
            yield Footer()

    @work
    async def action_write_marked(self) -> None:
        output_file = await self.app.push_screen_wait(FileChooserScreen())
        marked: set[VirtualDirEntry] = self.query_one(VirtualFileTree).marked
        self.log("writing marked", path=output_file)
        with output_file.open("w") as f:
            for data in marked:
                print(data.path, file=f)


class PathTreeApp(App):
    BINDINGS = [("q", "quit")]
    CSS_PATH = "file-copy.tcss"

    def __init__(self, paths: Iterable[str], limit: int | None = None):
        super().__init__()
        self._paths: Iterable[str] = paths
        self._limit: int | None = limit

    def _push_main_screen(self, root: VirtualDirEntry | None):
        if not root:
            return
        self.install_screen(PathTreeScreen(root), "main")
        self.push_screen("main")
        self.uninstall_screen("loading")

    def on_mount(self):
        self.install_screen(LoadingScreen(self._paths, self._limit), "loading")
        self.push_screen("loading", self._push_main_screen)

    def compose(self):
        yield Static("Loading failed")


def load_file(filename: str, prefix: str = "") -> Iterator[str]:
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line and line.startswith(prefix):
                yield line


def main():
    parser = argparse.ArgumentParser(description="Show a summary of files from rsync output")
    parser.add_argument("filename")
    parser.add_argument("--prefix")
    parser.add_argument("-n", "--limit", type=int)
    args = parser.parse_args()

    paths = list(load_file(args.filename, args.prefix))
    if args.limit:
        paths = paths[:args.limit]
    app = PathTreeApp(paths, len(paths))
    app.run()


if __name__ == "__main__":
    main()

