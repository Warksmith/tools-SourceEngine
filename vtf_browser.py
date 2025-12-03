#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Garry's Mod / Source Engine Material Browser (single-file version)

Requirements
------------
Python 3.8+
pip install PyQt5 vtf2img

- PyQt5 is used for the GUI.
- vtf2img is used to decode .vtf files into Pillow images internally.
  (https://github.com/julienc91/vtf2img)

Usage
-----
python main.py

Then:
- Click "File → Open Folder..." and choose a materials folder
  (e.g. <gmod>/garrysmod/materials).
- Thumbnails will appear in the right pane.
- Click a thumbnail to see details.
- Double-click a thumbnail to open the containing folder in your OS file explorer.

Notes
-----
- VMT parsing is intentionally simple and focuses on extracting $basetexture
  plus some common flags ($translucent, $selfillum).
- VTF decoding uses vtf2img. If it is missing, thumbnails will show a
  magenta/black checkerboard placeholder and the status bar will warn you.
- Scanning happens in a background QThread so the UI stays responsive.
- Thumbnails are generated in another background QThread after scanning.
"""

import os
import re
import sys
import io
import subprocess
import traceback
from dataclasses import dataclass, field
from typing import Dict, Optional, List

from PyQt5.QtCore import (
    Qt,
    QSize,
    QThread,
    pyqtSignal,
)
from PyQt5.QtGui import (
    QPixmap,
    QPainter,
    QColor,
    QIcon,
)
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QAction,
    QFileDialog,
    QStatusBar,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QListWidget,
    QListWidgetItem,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QDialog,
    QPushButton,
    QMessageBox,
)

# ---- VTF loader wrapper (vtf2img) ----------------------------------------


class VTFLoader:
    """Thin wrapper around vtf2img.Parser with simple caching.

    - load_thumbnail_bytes(): returns PNG bytes for a thumbnail and basic metadata.
    - get_full_pixmap(): returns a QPixmap for detail view plus metadata.
    """

    def __init__(self):
        try:
            from vtf2img import Parser  # type: ignore
        except ImportError:
            self.Parser = None
            self.is_available = False
        else:
            self.Parser = Parser
            self.is_available = True

        # Simple caches keyed by absolute path
        self._image_cache = {}  # path -> (PIL.Image, meta)

    def clear_cache(self):
        self._image_cache.clear()

    def _load_vtf(self, path: str):
        """Load VTF as a Pillow image with metadata."""
        if not self.is_available:
            raise RuntimeError("vtf2img is not available; install with 'pip install vtf2img'.")

        if path in self._image_cache:
            return self._image_cache[path]

        Parser = self.Parser
        parser = Parser(path)
        header = parser.header
        image = parser.get_image()  # Pillow Image

        meta = {
            "width": getattr(header, "width", None),
            "height": getattr(header, "height", None),
            "version": getattr(header, "version", None),
            "mode": getattr(image, "mode", None),
        }

        self._image_cache[path] = (image, meta)
        return image, meta

    def load_thumbnail_bytes(self, path: str, max_size: int = 128):
        """Return (png_bytes, meta) for a thumbnail of the VTF."""
        from PIL import Image  # vtf2img depends on Pillow; safe to import here

        image, meta = self._load_vtf(path)
        # Work on a copy to preserve cache
        img_copy = image.copy()
        # LANCZOS is good for thumbnails; fall back if missing
        resample = getattr(Image, "LANCZOS", getattr(Image, "BICUBIC", Image.BILINEAR))
        img_copy.thumbnail((max_size, max_size), resample=resample)

        buf = io.BytesIO()
        img_copy.save(buf, format="PNG")
        data = buf.getvalue()
        return data, meta

    def get_full_pixmap(self, path: str):
        """Return (QPixmap, meta) for full-size image (for detail view)."""
        image, meta = self._load_vtf(path)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        pix = QPixmap()
        pix.loadFromData(png_bytes)
        return pix, meta


# ---- Data model ----------------------------------------------------------


@dataclass
class MaterialEntry:
    """Represents a material or bare VTF entry in the UI."""

    id: int
    name: str
    directory: str  # relative to root, with forward slashes; '' for root
    vmt_path: Optional[str] = None
    vtf_path: Optional[str] = None
    base_texture: Optional[str] = None
    is_vmt: bool = False
    is_vtf: bool = False
    flags: Dict[str, str] = field(default_factory=dict)

    # Filled in when VTF is decoded:
    width: Optional[int] = None
    height: Optional[int] = None
    version: Optional[tuple] = None
    image_mode: Optional[str] = None

    # Runtime-only UI data:
    thumbnail: Optional[QPixmap] = None


# ---- VMT parsing helpers -------------------------------------------------


_VMT_KV_PATTERN = re.compile(
    r'^\s*("?)([^"\s]+)\1\s+"?([^"]+)"?\s*$', re.IGNORECASE
)


def strip_block_comments(text: str) -> str:
    """Remove /* ... */ comments from text."""
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def parse_vmt_file(path: str):
    """Parse a VMT file, focusing on $basetexture and a few common flags.

    Returns:
        (base_texture: Optional[str], flags: Dict[str, str])
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
    except Exception:
        print(f"[VMT] Error reading file: {path}")
        traceback.print_exc()
        return None, {}

    text = strip_block_comments(raw)
    flags: Dict[str, str] = {}
    base_texture = None

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("//") or line in ("{", "}"):
            continue

        m = _VMT_KV_PATTERN.match(line)
        if not m:
            continue

        key = m.group(2).lower()
        value = m.group(3).strip()

        # Normalize keys to keep $ prefix
        if not key.startswith("$"):
            key = "$" + key

        flags[key] = value

    # Prefer $basetexture if present
    base_texture = flags.get("$basetexture")
    return base_texture, flags


def normalize_base_texture_path(base_texture: str) -> str:
    """Convert a $basetexture value into a normalized key used for VTF lookup.

    - Lowercase
    - Strip 'materials/' prefix
    - Replace backslashes with forward slashes
    - Remove file extension if present
    """
    tex = base_texture.strip().replace("\\", "/")
    tex_lower = tex.lower()
    if tex_lower.startswith("materials/"):
        tex_lower = tex_lower[len("materials/") :]
    # Strip extension if present
    tex_lower = os.path.splitext(tex_lower)[0]
    return tex_lower


# ---- Background scanning thread -----------------------------------------


class ScanThread(QThread):
    """Recursively scan a root folder for VMT/VTF files and build entries."""

    materialFound = pyqtSignal(object)  # MaterialEntry
    progress = pyqtSignal(str)
    finishedScanning = pyqtSignal()

    def __init__(self, root_path: str, parent=None):
        super().__init__(parent)
        self.root_path = root_path

    def run(self):
        try:
            self._run_impl()
        except Exception:
            print("[ScanThread] Unhandled exception during scan:")
            traceback.print_exc()
        finally:
            self.finishedScanning.emit()

    def _run_impl(self):
        root = self.root_path
        vtf_files: Dict[str, str] = {}
        vmt_files: List[str] = []

        self.progress.emit(f"Scanning for .vtf and .vmt under: {root}")

        # First pass: gather all VTF and VMT file paths
        for dirpath, dirnames, filenames in os.walk(root):
            if self.isInterruptionRequested():
                return
            for fname in filenames:
                lower = fname.lower()
                full_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(full_path, root)
                if lower.endswith(".vtf"):
                    key = os.path.splitext(rel_path.replace("\\", "/"))[0].lower()
                    vtf_files[key] = full_path
                elif lower.endswith(".vmt"):
                    vmt_files.append(full_path)

        self.progress.emit(
            f"Found {len(vmt_files)} VMTs and {len(vtf_files)} VTFs. Parsing VMTs..."
        )

        next_id = 1
        used_vtf_keys = set()

        # Second pass: parse VMTs and map to VTFs
        for vmt_path in vmt_files:
            if self.isInterruptionRequested():
                return

            base_texture, flags = parse_vmt_file(vmt_path)
            vtf_path = None
            if base_texture:
                key = normalize_base_texture_path(base_texture)
                if key in vtf_files:
                    vtf_path = vtf_files[key]
                    used_vtf_keys.add(key)

            rel_vmt = os.path.relpath(vmt_path, root).replace("\\", "/")
            directory = os.path.dirname(rel_vmt)
            name = os.path.splitext(os.path.basename(vmt_path))[0]

            entry = MaterialEntry(
                id=next_id,
                name=name,
                directory=directory,
                vmt_path=vmt_path,
                vtf_path=vtf_path,
                base_texture=base_texture,
                is_vmt=True,
                is_vtf=bool(vtf_path),
                flags=flags,
            )
            next_id += 1
            self.materialFound.emit(entry)

        self.progress.emit("Adding orphan VTF files (not referenced by VMT)...")

        # Third pass: add VTF-only entries (no corresponding VMT)
        for key, vtf_path in vtf_files.items():
            if self.isInterruptionRequested():
                return
            if key in used_vtf_keys:
                continue

            rel_vtf = os.path.relpath(vtf_path, root).replace("\\", "/")
            directory = os.path.dirname(rel_vtf)
            name = os.path.splitext(os.path.basename(vtf_path))[0]

            entry = MaterialEntry(
                id=next_id,
                name=name,
                directory=directory,
                vmt_path=None,
                vtf_path=vtf_path,
                base_texture=None,
                is_vmt=False,
                is_vtf=True,
                flags={},
            )
            next_id += 1
            self.materialFound.emit(entry)

        self.progress.emit("Scanning complete.")


# ---- Thumbnail generation thread ----------------------------------------


class ThumbnailThread(QThread):
    """Decode VTFs into thumbnails in a background thread."""

    thumbnailReady = pyqtSignal(int, bytes, dict)  # entry_id, PNG bytes, meta dict
    progress = pyqtSignal(str)

    def __init__(self, materials: List[MaterialEntry], loader: VTFLoader, max_size=128, parent=None):
        super().__init__(parent)
        self.materials = materials
        self.loader = loader
        self.max_size = max_size

    def run(self):
        if not self.loader.is_available:
            self.progress.emit("VTF decoding disabled: vtf2img is not installed.")
            return

        total = len(self.materials)
        for idx, entry in enumerate(self.materials, start=1):
            if self.isInterruptionRequested():
                return
            if not entry.vtf_path:
                continue
            try:
                data, meta = self.loader.load_thumbnail_bytes(entry.vtf_path, self.max_size)
            except Exception as exc:
                print(f"[ThumbnailThread] Failed to load VTF '{entry.vtf_path}': {exc}")
                traceback.print_exc()
                continue

            self.thumbnailReady.emit(entry.id, data, meta)
            if idx % 20 == 0 or idx == total:
                self.progress.emit(f"Generated thumbnails: {idx}/{total}")


# ---- GUI helpers ---------------------------------------------------------


def create_checkerboard_pixmap(size: int = 128, square: int = 16):
    """Create a magenta/black checkerboard missing-texture placeholder."""
    pix = QPixmap(size, size)
    pix.fill(QColor(255, 0, 255))
    painter = QPainter(pix)
    color1 = QColor(255, 0, 255)
    color2 = QColor(0, 0, 0)

    toggle_row = False
    for y in range(0, size, square):
        toggle_row = not toggle_row
        toggle = toggle_row
        for x in range(0, size, square):
            painter.fillRect(x, y, square, square, color1 if toggle else color2)
            toggle = not toggle
    painter.end()
    return pix


def create_loading_pixmap(size: int = 128):
    """Simple gray placeholder while thumbnail is loading."""
    pix = QPixmap(size, size)
    pix.fill(QColor(80, 80, 80))
    painter = QPainter(pix)
    painter.setPen(QColor(180, 180, 180))
    painter.drawText(pix.rect(), Qt.AlignCenter, "Loading...")
    painter.end()
    return pix


def open_in_file_explorer(path: str):
    """Open the folder containing the given path in the OS file explorer."""
    if not path or not os.path.exists(path):
        return

    folder = os.path.dirname(os.path.abspath(path))
    if sys.platform.startswith("win"):
        os.startfile(folder)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", folder])
    else:
        subprocess.Popen(["xdg-open", folder])


# ---- Detail dialog -------------------------------------------------------


class DetailDialog(QDialog):
    """Shows a larger preview and metadata for a MaterialEntry."""

    def __init__(self, material: MaterialEntry, loader: VTFLoader, placeholder: QPixmap, parent=None):
        super().__init__(parent)
        self.material = material
        self.loader = loader
        self.placeholder = placeholder
        self.setWindowTitle(f"Material Details - {material.name}")
        self.resize(500, 400)

        main_layout = QVBoxLayout(self)

        # Preview
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.preview_label, stretch=2)

        # Info labels
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        self.path_label = QLabel()
        self.meta_label = QLabel()
        self.flags_label = QLabel()
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.meta_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.flags_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        info_layout.addWidget(self.path_label)
        info_layout.addWidget(self.meta_label)
        info_layout.addWidget(self.flags_label)
        main_layout.addWidget(info_widget, stretch=1)

        # Buttons
        btn_widget = QWidget()
        btn_layout = QHBoxLayout(btn_widget)
        btn_layout.addStretch(1)
        self.open_folder_btn = QPushButton("Open Containing Folder")
        self.close_btn = QPushButton("Close")
        btn_layout.addWidget(self.open_folder_btn)
        btn_layout.addWidget(self.close_btn)
        main_layout.addWidget(btn_widget)

        self.close_btn.clicked.connect(self.accept)
        self.open_folder_btn.clicked.connect(self.open_folder)

        self._populate()

    def _populate(self):
        # Preview: try to load full pixmap; fall back to thumbnail; then placeholder.
        pix = None
        if self.material.vtf_path and self.loader.is_available:
            try:
                pix, meta = self.loader.get_full_pixmap(self.material.vtf_path)
                # Update meta if missing
                if self.material.width is None:
                    self.material.width = meta.get("width")
                    self.material.height = meta.get("height")
                    self.material.version = meta.get("version")
                    self.material.image_mode = meta.get("mode")
            except Exception:
                traceback.print_exc()
                pix = None

        if pix is None and self.material.thumbnail is not None:
            pix = self.material.thumbnail

        if pix is None:
            pix = self.placeholder

        self.preview_label.setPixmap(pix.scaled(256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # Paths & metadata
        vmt_path = self.material.vmt_path or "<none>"
        vtf_path = self.material.vtf_path or "<missing>"

        path_text = f"<b>VMT:</b> {vmt_path}<br><b>VTF:</b> {vtf_path}"
        self.path_label.setText(path_text)

        w = self.material.width
        h = self.material.height
        version = self.material.version
        mode = self.material.image_mode

        meta_lines = []
        if w and h:
            meta_lines.append(f"Resolution: {w} x {h}")
        if version:
            meta_lines.append(f"VTF version: {version}")
        if mode:
            meta_lines.append(f"Image mode: {mode}")
        if self.material.base_texture:
            meta_lines.append(f"$basetexture: {self.material.base_texture}")
        if not meta_lines:
            meta_lines.append("No VTF metadata available.")
        self.meta_label.setText("<br>".join(meta_lines))

        flag_lines = []
        if self.material.flags:
            # Show a few interesting flags first
            interesting = ["$translucent", "$selfillum", "$alphatest"]
            for k in interesting:
                if k in self.material.flags:
                    flag_lines.append(f"{k} = {self.material.flags[k]}")
            # Add remaining keys
            for k, v in self.material.flags.items():
                if k in interesting:
                    continue
                flag_lines.append(f"{k} = {v}")
        else:
            flag_lines.append("No additional VMT flags parsed.")
        self.flags_label.setText("<br>".join(flag_lines))

    def open_folder(self):
        path = self.material.vmt_path or self.material.vtf_path
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Open Folder", "File no longer exists on disk.")
            return
        open_in_file_explorer(path)


# ---- Main window ---------------------------------------------------------


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Source / Garry's Mod Material Browser (PyQt)")
        self.resize(1200, 700)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Data
        self.current_root: Optional[str] = None
        self.materials: Dict[int, MaterialEntry] = {}
        self.scan_thread: Optional[ScanThread] = None
        self.thumb_thread: Optional[ThumbnailThread] = None

        self.vtf_loader = VTFLoader()
        self.placeholder_missing = create_checkerboard_pixmap(128, 16)
        self.placeholder_loading = create_loading_pixmap(128)

        # Filtering state
        self.current_dir_filter = ""  # relative directory; '' = all
        self.search_text = ""
        self.type_filter = "All"  # "All", "VMT only", "VTF only"

        # Directory tree mapping: rel_dir -> QTreeWidgetItem
        self.dir_items: Dict[str, QTreeWidgetItem] = {}

        self._init_menu()
        self._init_ui()

        if not self.vtf_loader.is_available:
            self.status_bar.showMessage(
                "vtf2img not installed - VTF decoding disabled (run: pip install vtf2img)",
                10000,
            )

    # ---- UI setup ----

    def _init_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")

        open_action = QAction("&Open Folder...", self)
        open_action.triggered.connect(self.open_folder)
        file_menu.addAction(open_action)

        refresh_action = QAction("&Refresh", self)
        refresh_action.triggered.connect(self.refresh_scan)
        file_menu.addAction(refresh_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _init_ui(self):
        splitter = QSplitter(Qt.Horizontal, self)
        self.setCentralWidget(splitter)

        # Left: directory tree
        self.dir_tree = QTreeWidget()
        self.dir_tree.setHeaderHidden(True)
        self.dir_tree.itemClicked.connect(self.on_dir_clicked)
        splitter.addWidget(self.dir_tree)

        # Right: filter bar + thumbnail list
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        filter_widget = QWidget()
        filter_layout = QHBoxLayout(filter_widget)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.addWidget(QLabel("Filter:"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Search by name or path...")
        self.filter_edit.textChanged.connect(self.on_filter_text_changed)
        filter_layout.addWidget(self.filter_edit)

        self.type_combo = QComboBox()
        self.type_combo.addItems(["All", "VMT only", "VTF only"])
        self.type_combo.currentTextChanged.connect(self.on_type_filter_changed)
        filter_layout.addWidget(QLabel("Type:"))
        filter_layout.addWidget(self.type_combo)

        right_layout.addWidget(filter_widget)

        self.thumb_list = QListWidget()
        self.thumb_list.setViewMode(QListWidget.IconMode)
        self.thumb_list.setResizeMode(QListWidget.Adjust)
        self.thumb_list.setMovement(QListWidget.Static)
        self.thumb_list.setIconSize(QSize(128, 128))
        self.thumb_list.setSpacing(8)
        self.thumb_list.setSelectionMode(QListWidget.SingleSelection)
        self.thumb_list.itemClicked.connect(self.on_item_clicked)
        self.thumb_list.itemDoubleClicked.connect(self.on_item_double_clicked)

        right_layout.addWidget(self.thumb_list)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 1)

    # ---- Folder handling & scanning ----

    def open_folder(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Materials Root Folder",
            self.current_root or os.getcwd(),
        )
        if not path:
            return
        self.start_scan(path)

    def refresh_scan(self):
        if not self.current_root:
            QMessageBox.information(self, "Refresh", "No folder selected yet.")
            return
        self.start_scan(self.current_root)

    def start_scan(self, root_path: str):
        # Stop any existing threads
        if self.scan_thread and self.scan_thread.isRunning():
            self.scan_thread.requestInterruption()
            self.scan_thread.wait(1000)
        if self.thumb_thread and self.thumb_thread.isRunning():
            self.thumb_thread.requestInterruption()
            self.thumb_thread.wait(1000)

        self.vtf_loader.clear_cache()

        # Clear data & UI
        self.current_root = root_path
        self.materials.clear()
        self.thumb_list.clear()
        self.dir_tree.clear()
        self.dir_items.clear()
        self.current_dir_filter = ""
        self.search_text = ""
        self.filter_edit.setText("")

        # Root item
        root_name = os.path.basename(root_path.rstrip(os.sep)) or root_path
        root_item = QTreeWidgetItem([root_name])
        root_item.setData(0, Qt.UserRole, "")
        self.dir_tree.addTopLevelItem(root_item)
        self.dir_tree.expandItem(root_item)
        self.dir_items[""] = root_item

        self.status_bar.showMessage(f"Scanning: {root_path} ...")

        # Start scanning thread
        self.scan_thread = ScanThread(root_path)
        self.scan_thread.materialFound.connect(self.on_material_found)
        self.scan_thread.progress.connect(self.on_scan_progress)
        self.scan_thread.finishedScanning.connect(self.on_scan_finished)
        self.scan_thread.start()

    def on_scan_progress(self, message: str):
        self.status_bar.showMessage(message)

    def on_material_found(self, entry_obj):
        # Called from scan thread; entry_obj is a MaterialEntry
        entry: MaterialEntry = entry_obj
        self.materials[entry.id] = entry
        self._ensure_directory_item(entry.directory)
        # Add to list if passes current filters (initially they do)
        if self._entry_matches_filters(entry):
            self._add_list_item_for_entry(entry)

    def on_scan_finished(self):
        self.status_bar.showMessage("Scan complete. Generating thumbnails...")
        # Start thumbnail generation over all materials
        if not self.materials:
            self.status_bar.showMessage("Scan complete. No materials found.")
            return
        materials = list(self.materials.values())
        self.thumb_thread = ThumbnailThread(materials, self.vtf_loader, max_size=128)
        self.thumb_thread.thumbnailReady.connect(self.on_thumbnail_ready)
        self.thumb_thread.progress.connect(self.on_thumb_progress)
        self.thumb_thread.finished.connect(
            lambda: self.status_bar.showMessage("All thumbnails generated.")
        )
        self.thumb_thread.start()

    def on_thumb_progress(self, message: str):
        self.status_bar.showMessage(message)

    def on_thumbnail_ready(self, entry_id: int, png_bytes: bytes, meta: dict):
        entry = self.materials.get(entry_id)
        if not entry:
            return

        pix = QPixmap()
        pix.loadFromData(png_bytes)
        entry.thumbnail = pix
        entry.width = meta.get("width")
        entry.height = meta.get("height")
        entry.version = meta.get("version")
        entry.image_mode = meta.get("mode")

        # Update any visible list items with this ID
        for i in range(self.thumb_list.count()):
            item = self.thumb_list.item(i)
            if item.data(Qt.UserRole) == entry_id:
                item.setIcon(QIcon(pix))
                break

    # ---- Directory tree ----

    def _ensure_directory_item(self, rel_dir: str):
        """Ensure that the directory entry exists in the tree."""
        if rel_dir in self.dir_items:
            return

        # Build ancestors first
        parts = [p for p in rel_dir.replace("\\", "/").split("/") if p]
        path_so_far = ""
        parent_item = self.dir_items[""]
        for part in parts:
            path_so_far = part if not path_so_far else path_so_far + "/" + part
            if path_so_far not in self.dir_items:
                item = QTreeWidgetItem([part])
                item.setData(0, Qt.UserRole, path_so_far)
                parent_item.addChild(item)
                self.dir_items[path_so_far] = item
            parent_item = self.dir_items[path_so_far]

    def on_dir_clicked(self, item: QTreeWidgetItem, _column: int):
        rel = item.data(0, Qt.UserRole)
        self.current_dir_filter = rel or ""
        self.refresh_list_from_filters()

    # ---- Filtering & list management ----

    def on_filter_text_changed(self, text: str):
        self.search_text = text.strip().lower()
        self.refresh_list_from_filters()

    def on_type_filter_changed(self, text: str):
        self.type_filter = text
        self.refresh_list_from_filters()

    def _entry_matches_filters(self, entry: MaterialEntry) -> bool:
        # Directory filter: show entries under selected directory (including subdirs)
        if self.current_dir_filter:
            if entry.directory == self.current_dir_filter:
                pass
            elif entry.directory.startswith(self.current_dir_filter + "/"):
                pass
            else:
                return False

        # Type filter
        if self.type_filter == "VMT only":
            if not entry.vmt_path:
                return False
        elif self.type_filter == "VTF only":
            if entry.vmt_path:
                return False

        # Text filter
        if self.search_text:
            haystack = [
                entry.name.lower(),
                (entry.base_texture or "").lower(),
                (entry.vmt_path or "").lower(),
                (entry.vtf_path or "").lower(),
            ]
            if not any(self.search_text in s for s in haystack):
                return False

        return True

    def refresh_list_from_filters(self):
        self.thumb_list.clear()
        for entry in self.materials.values():
            if self._entry_matches_filters(entry):
                self._add_list_item_for_entry(entry)

    def _add_list_item_for_entry(self, entry: MaterialEntry):
        text = entry.name
        if entry.vmt_path and not entry.vtf_path:
            text += "\n(missing VTF)"
        elif not entry.vmt_path and entry.vtf_path:
            text += "\n(VTF only)"

        item = QListWidgetItem(text)
        item.setData(Qt.UserRole, entry.id)
        # Set initial icon
        if entry.thumbnail is not None:
            item.setIcon(QIcon(entry.thumbnail))
        elif entry.vtf_path and self.vtf_loader.is_available:
            item.setIcon(QIcon(self.placeholder_loading))
        else:
            item.setIcon(QIcon(self.placeholder_missing))

        self.thumb_list.addItem(item)

    # ---- Item interactions ----

    def _get_entry_for_item(self, item: QListWidgetItem) -> Optional[MaterialEntry]:
        entry_id = item.data(Qt.UserRole)
        return self.materials.get(entry_id)

    def on_item_clicked(self, item: QListWidgetItem):
        entry = self._get_entry_for_item(item)
        if not entry:
            return
        dlg = DetailDialog(entry, self.vtf_loader, self.placeholder_missing, self)
        dlg.exec_()

    def on_item_double_clicked(self, item: QListWidgetItem):
        entry = self._get_entry_for_item(item)
        if not entry:
            return
        path = entry.vmt_path or entry.vtf_path
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Open Folder", "File no longer exists on disk.")
            return
        open_in_file_explorer(path)


# ---- main entry point ----------------------------------------------------


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
