#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Source Engine MDL Browser (single-file, PyQt5)

What it does
------------
- Desktop GUI to browse Source Engine model (.mdl) files.
- Lets you choose a root folder (game root or models folder).
- Recursively finds all *.mdl files.
- Parses the MDL header (studiohdr_t) to extract:
    * Format ID ("IDST"), version, checksum
    * Internal model name
    * File length
    * Eye position (vector)
    * Hull and view bounding boxes (min/max vectors)
    * Flags
    * Bone / sequence / texture / bodypart counts, etc.
- Shows a directory tree on the left, and a grid of model thumbnails on the right.
- Clicking a thumbnail opens a detail dialog with the metadata above.

Requirements
------------
- Python 3.8+
- PyQt5

Install:
    pip install PyQt5

Usage
-----
    python mdl_browser.py

Then:
- File -> Open Folder...
- Select a folder containing MDL files (e.g. a "models" folder, or the root
  of an extracted GMod addon).
- Models will be indexed and displayed in a thumbnail grid.

Notes
-----
- This only parses the *header* of MDL files as documented in Valve's
  studiohdr_t struct (Valve Developer Wiki), not geometry or animations.
- Thumbnails are generic icons, not rendered 3D previews.
- The code is intentionally monolithic but structured into classes to keep
  it readable and hackable.
"""

import os
import struct
import sys
import traceback
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

from PyQt5.QtCore import (
    Qt,
    QSize,
    QThread,
    pyqtSignal,
    QEvent,
)
from PyQt5.QtGui import (
    QPixmap,
    QPainter,
    QColor,
    QPen,
    QIcon,
    QFont,
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
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QDialog,
    QPushButton,
    QMessageBox,
)

try:
    from PyQt5.QtWidgets import QOpenGLWidget
except Exception:
    QOpenGLWidget = None


# ---------------------------------------------------------------------------
# MDL header parsing
# ---------------------------------------------------------------------------

# Little-endian header layout based on Valve's studiohdr_t docs:
# https://developer.valvesoftware.com/wiki/MDL
#
# We parse:
#   int id;
#   int version;
#   int checksum;
#   char name[64];
#   int length;
#   Vector eyePosition;    // 3 floats
#   Vector illumposition;  // 3 floats
#   Vector hull_min;       // 3 floats
#   Vector hull_max;       // 3 floats
#   Vector view_bbmin;     // 3 floats
#   Vector view_bbmax;     // 3 floats
#   int flags;
#   int numbones;
#   int boneindex;
#   int numbonecontrollers;
#   int bonecontrollerindex;
#   int numhitboxsets;
#   int hitboxsetindex;
#   int numlocalanim;
#   int localanimindex;
#   int numlocalseq;
#   int localseqindex;
#   int activitylistversion;
#   int eventsindexed;
#   int numtextures;
#   int textureindex;
#   int numcdtextures;
#   int cdtextureindex;
#   int numskinref;
#   int numskinfamilies;
#   int skinindex;
#   int numbodyparts;
#   int bodypartindex;
#
# Total parsed bytes: 228.


MDL_MAGIC_IDST = int.from_bytes(b'IDST', 'little')

# 6 vectors: eye, illum, hull_min, hull_max, view_bbmin, view_bbmax = 18 floats
# 22 ints after that
MDL_HEADER_FMT = '<iii64si' + 'f' * (3 * 6) + 'i' * 22
MDL_HEADER_SIZE = struct.calcsize(MDL_HEADER_FMT)

@dataclass
class MDLHeader:
    """Minimal representation of an MDL header."""
    magic: int
    version: int
    checksum: int
    internal_name: str
    length: int

    eye_pos: Tuple[float, float, float]
    hull_min: Tuple[float, float, float]
    hull_max: Tuple[float, float, float]
    view_bbmin: Tuple[float, float, float]
    view_bbmax: Tuple[float, float, float]

    flags: int
    numbones: int
    numlocalanim: int
    numlocalseq: int
    numtextures: int
    numbodyparts: int
    # NEW: texture metadata
    texture_names: List[str]
    cdtexture_paths: List[str]


def parse_mdl_header(path: str) -> Optional[MDLHeader]:
    """Parse the minimal MDL header needed for metadata + texture info.

    Returns:
        MDLHeader instance, or None if parsing fails or magic/version look wrong.
    """
    try:
        with open(path, 'rb') as f:
            header_data = f.read(MDL_HEADER_SIZE)
            if len(header_data) < MDL_HEADER_SIZE:
                return None
            try:
                unpacked = struct.unpack(MDL_HEADER_FMT, header_data)
            except struct.error:
                return None
            (
                magic,
                version,
                checksum,
                raw_name,
                length,
                eye_x, eye_y, eye_z,
                illum_x, illum_y, illum_z,
                hull_min_x, hull_min_y, hull_min_z,
                hull_max_x, hull_max_y, hull_max_z,
                view_min_x, view_min_y, view_min_z,
                view_max_x, view_max_y, view_max_z,
                flags,
                numbones,
                boneindex,
                numbonecontrollers,
                bonecontrollerindex,
                numhitboxsets,
                hitboxsetindex,
                numlocalanim,
                localanimindex,
                numlocalseq,
                localseqindex,
                activitylistversion,
                eventsindexed,
                numtextures,
                textureindex,
                numcdtextures,
                cdtextureindex,
                numskinref,
                numskinfamilies,
                skinindex,
                numbodyparts,
                bodypartindex,
            ) = unpacked

            # Sanity checks
            if magic != MDL_MAGIC_IDST:
                # Not a Source MDL we understand
                return None

            # Very old or very new versions might not match; you can relax/tighten this if needed.
            if not (40 <= version <= 54):  # Common Source MDL versions range
                # Still accept; just don't bail.
                pass

            internal_name = raw_name.split(b'\x00', 1)[0].decode('ascii', errors='ignore')

            # Determine file size for bounds checking
            f.seek(0, os.SEEK_END)
            file_size = f.tell()

            texture_names: List[str] = []
            cdtexture_paths: List[str] = []

            # --- Parse texture names (mstudiotexture_t array) ---
            # MDL stores an array of texture structs at 'textureindex'.
            # The first int of each struct is sznameindex (offset from the
            # struct to the actual C-string name).
            # Struct size is 15 ints in Valve's docs (60 bytes).
            # We only need the first int (sznameindex) but must use the
            # correct struct stride so subsequent entries are read correctly.
            TEXTURE_STRUCT_SIZE = 60
            if (
                numtextures > 0
                and textureindex > 0
                and textureindex < file_size
            ):
                for i in range(numtextures):
                    base = textureindex + i * TEXTURE_STRUCT_SIZE
                    if base + 4 > file_size:
                        break
                    f.seek(base)
                    sz_bytes = f.read(4)
                    if len(sz_bytes) < 4:
                        break
                    (sznameindex,) = struct.unpack('<i', sz_bytes)

                    # Guard against obviously bogus name offsets
                    if sznameindex <= 0 or sznameindex > 4096:
                        continue
                    name_offset = base + sznameindex
                    if name_offset <= 0 or name_offset >= file_size:
                        continue
                    f.seek(name_offset)
                    name_bytes = []
                    while True:
                        b = f.read(1)
                        if not b or b == b'\x00':
                            break
                        name_bytes.append(b)
                        if len(name_bytes) > 512:
                            break
                    if not name_bytes:
                        continue
                    name = b''.join(name_bytes).decode('ascii', errors='ignore').strip()
                    if name and name not in texture_names:
                        texture_names.append(name)

            # --- Parse CD texture search paths ---
            # At cdtextureindex there is an array of int offsets, each pointing
            # to a C-string path relative to "materials/".
            if (
                numcdtextures > 0
                and cdtextureindex > 0
                and cdtextureindex < file_size
            ):
                # cdtextureindex points to an array of int offsets (relative
                # to the start of the header/file). Each int is itself an
                # offset from the header base to a null-terminated string.
                f.seek(cdtextureindex)
                for i in range(numcdtextures):
                    off_bytes = f.read(4)
                    if len(off_bytes) < 4:
                        break
                    (str_offset,) = struct.unpack('<i', off_bytes)
                    if str_offset == 0:
                        continue
                    # str_offset is relative to header base (file start), not
                    # relative to cdtextureindex.
                    path_pos = str_offset
                    if path_pos <= 0 or path_pos >= file_size:
                        continue
                    f.seek(path_pos)
                    path_bytes = []
                    while True:
                        b = f.read(1)
                        if not b or b == b'\x00':
                            break
                        path_bytes.append(b)
                        if len(path_bytes) > 512:
                            break
                    if not path_bytes:
                        continue
                    path = b''.join(path_bytes).decode('ascii', errors='ignore').strip()
                    if path and path not in cdtexture_paths:
                        cdtexture_paths.append(path)
    except OSError:
        return None

    return MDLHeader(
        magic=magic,
        version=version,
        checksum=checksum,
        internal_name=internal_name,
        length=length,
        eye_pos=(eye_x, eye_y, eye_z),
        hull_min=(hull_min_x, hull_min_y, hull_min_z),
        hull_max=(hull_max_x, hull_max_y, hull_max_z),
        view_bbmin=(view_min_x, view_min_y, view_min_z),
        view_bbmax=(view_max_x, view_max_y, view_max_z),
        flags=flags,
        numbones=numbones,
        numlocalanim=numlocalanim,
        numlocalseq=numlocalseq,
        numtextures=numtextures,
        numbodyparts=numbodyparts,
        texture_names=texture_names,
        cdtexture_paths=cdtexture_paths,
    )


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ModelEntry:
    """Represents a single MDL file in the UI."""
    id: int
    name: str                 # file name without extension
    directory: str            # relative to root, with forward slashes; '' for root
    mdl_path: str             # absolute path

    # Header metadata
    magic_ok: bool
    version: Optional[int] = None
    checksum: Optional[int] = None
    internal_name: Optional[str] = None
    file_length: Optional[int] = None
    file_size: Optional[int] = None

    eye_pos: Optional[Tuple[float, float, float]] = None
    hull_min: Optional[Tuple[float, float, float]] = None
    hull_max: Optional[Tuple[float, float, float]] = None
    view_bbmin: Optional[Tuple[float, float, float]] = None
    view_bbmax: Optional[Tuple[float, float, float]] = None

    flags: Optional[int] = None
    numbones: Optional[int] = None
    numlocalanim: Optional[int] = None
    numlocalseq: Optional[int] = None
    numtextures: Optional[int] = None
    numbodyparts: Optional[int] = None

    # NEW: textures
    texture_names: Optional[List[str]] = None
    cdtexture_paths: Optional[List[str]] = None

    # Companion files & lazy-loaded mesh
    vvd_path: Optional[str] = None
    vtx_path: Optional[str] = None
    mesh_loaded: bool = False
    mesh_error: Optional[str] = None
    vertices: Optional[List[Tuple[float, float, float]]] = None
    normals: Optional[List[Tuple[float, float, float]]] = None
    uvs: Optional[List[Tuple[float, float]]] = None
    indices: Optional[List[int]] = None
    # UI runtime data
    icon: Optional[QPixmap] = None


# ---------------------------------------------------------------------------
# Background scanning thread
# ---------------------------------------------------------------------------

class ScanThread(QThread):
    """Recursively scans a root folder for .mdl files and parses headers."""

    modelFound = pyqtSignal(object)  # ModelEntry
    progress = pyqtSignal(str)
    finishedScanning = pyqtSignal()

    def __init__(self, root_path: str, parent=None):
        super().__init__(parent)
        self.root_path = root_path

    def run(self):
        try:
            self._run_impl()
        except Exception:
            print("[ScanThread] Unhandled exception:")
            traceback.print_exc()
        finally:
            self.finishedScanning.emit()

    def _run_impl(self):
        root = self.root_path
        self.progress.emit(f"Scanning for .mdl files under: {root}")

        next_id = 1

        for dirpath, dirnames, filenames in os.walk(root):
            if self.isInterruptionRequested():
                return

            for fname in filenames:
                if self.isInterruptionRequested():
                    return

                if not fname.lower().endswith('.mdl'):
                    continue

                full_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(full_path, root).replace('\\', '/')
                directory = os.path.dirname(rel_path)
                name_no_ext = os.path.splitext(os.path.basename(fname))[0]

                file_size = None
                try:
                    file_size = os.path.getsize(full_path)
                except OSError:
                    pass

                header = parse_mdl_header(full_path)
                # If srctools is available, prefer its texture enumeration
                # (more accurate for models in a game root). Fall back to
                # header parsing when srctools is not present or fails.
                tex_names = []
                try:
                    tex_names = extract_textures_with_srctools(self.root_path, full_path)
                except Exception:
                    tex_names = []
                # If srctools found nothing, keep header's parsed names (if any)
                if not tex_names and header is not None:
                    tex_names = header.texture_names
                # companion file detection for potential mesh data
                base_no_ext = os.path.splitext(full_path)[0]
                vvd_path = base_no_ext + '.vvd'
                vtx_path = None
                for ext in ['.dx90.vtx', '.vtx']:
                    cand = base_no_ext + ext
                    if os.path.exists(cand):
                        vtx_path = cand
                        break
                if header is None:
                    # Probably not a valid Source MDL, but still list it.
                    entry = ModelEntry(
                        id=next_id,
                        name=name_no_ext,
                        directory=directory,
                        mdl_path=full_path,
                        magic_ok=False,
                        file_size=file_size,
                    )
                else:
                    entry = ModelEntry(
                        id=next_id,
                        name=name_no_ext,
                        directory=directory,
                        mdl_path=full_path,
                        magic_ok=True,
                        version=header.version,
                        checksum=header.checksum,
                        internal_name=header.internal_name,
                        file_length=header.length,
                        file_size=file_size,
                        eye_pos=header.eye_pos,
                        hull_min=header.hull_min,
                        hull_max=header.hull_max,
                        view_bbmin=header.view_bbmin,
                        view_bbmax=header.view_bbmax,
                        flags=header.flags,
                        numbones=header.numbones,
                        numlocalanim=header.numlocalanim,
                        numlocalseq=header.numlocalseq,
                        numtextures=header.numtextures,
                        numbodyparts=header.numbodyparts,
                        texture_names=tex_names,
                        cdtexture_paths=header.cdtexture_paths,
                        # companion files
                        vvd_path=vvd_path if os.path.exists(vvd_path) else vvd_path,
                        vtx_path=vtx_path,
                    )

                next_id += 1
                self.modelFound.emit(entry)

        self.progress.emit("Scanning complete.")


# ---------------------------------------------------------------------------
# GUI helpers
# ---------------------------------------------------------------------------

def create_model_icon(size: int = 128) -> QPixmap:
    """Create a simple generic '3D box' icon for MDLs."""
    pix = QPixmap(size, size)
    pix.fill(QColor(40, 40, 40))

    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing, True)

    # Draw cube-ish wireframe
    pen = QPen(QColor(220, 220, 220))
    pen.setWidth(2)
    painter.setPen(pen)

    margin = int(size * 0.2)
    depth = int(size * 0.18)

    # Front rectangle
    x1, y1 = margin, margin + depth
    x2, y2 = size - margin, size - margin
    painter.drawRect(x1, y1, x2 - x1, y2 - y1)

    # Back rectangle (offset up-left)
    bx1, by1 = x1 - depth, y1 - depth
    bx2, by2 = x2 - depth, y2 - depth
    painter.drawRect(bx1, by1, bx2 - bx1, by2 - by1)

    # Connect corners
    painter.drawLine(x1, y1, bx1, by1)
    painter.drawLine(x2, y1, bx2, by1)
    painter.drawLine(x1, y2, bx1, by2)
    painter.drawLine(x2, y2, bx2, by2)

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
        import subprocess
        subprocess.Popen(["open", folder])
    else:
        import subprocess
        subprocess.Popen(["xdg-open", folder])


# ---------------------------------------------------------------------------
# Mesh loader & OpenGL preview (Phase 1)
# ---------------------------------------------------------------------------

class ModelMeshLoader:
    """Helper to produce a simple triangle mesh for preview.

    Currently this produces a box mesh based on the header hull min/max as
    a reliable fallback. If `srctools` integration is desired, add it here
    to parse `.vvd`/`.vtx` into vertex/index arrays and return them.
    """

    @staticmethod
    def load_mesh(mdl_path: str, vvd_path: Optional[str], vtx_path: Optional[str], model_entry: Optional['ModelEntry']):
        # Try to use hull bounds from the model entry as a basic representative mesh
        if model_entry and getattr(model_entry, 'hull_min', None) and getattr(model_entry, 'hull_max', None):
            min_x, min_y, min_z = model_entry.hull_min
            max_x, max_y, max_z = model_entry.hull_max
        else:
            # Fallback cube
            min_x, min_y, min_z = (-8.0, -8.0, -8.0)
            max_x, max_y, max_z = (8.0, 8.0, 8.0)

        # 8 vertices of a box
        vertices = [
            (min_x, min_y, min_z),
            (max_x, min_y, min_z),
            (max_x, max_y, min_z),
            (min_x, max_y, min_z),
            (min_x, min_y, max_z),
            (max_x, min_y, max_z),
            (max_x, max_y, max_z),
            (min_x, max_y, max_z),
        ]

        # normals per vertex (approx)
        normals = [(0.0, 0.0, 1.0)] * len(vertices)

        # simple UVs
        uvs = [(0.0, 0.0)] * len(vertices)

        # 12 triangles (two per face)
        indices = [
            0, 1, 2, 0, 2, 3,
            4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4,
            2, 3, 7, 2, 7, 6,
            1, 2, 6, 1, 6, 5,
            3, 0, 4, 3, 4, 7,
        ]

        return vertices, normals, uvs, indices


class ModelPreviewWidget(QOpenGLWidget if QOpenGLWidget is not None else object):
    """Simple OpenGL preview widget for a single `ModelEntry`.

    Uses legacy immediate-mode drawing (glBegin/glEnd) for a quick Phase 1
    implementation. If PyOpenGL is missing the widget will display an
    informative message instead.
    """

    def __init__(self, model: ModelEntry, parent=None):
        if QOpenGLWidget is None:
            # Dummy object when GL widget unavailable
            return
        super().__init__(parent)
        self.model = model
        self.rot_y = 30.0
        self.distance = 0.0
        self._gl_ok = False
        self._last_x = None

    def initializeGL(self):
        try:
            from OpenGL.GL import glEnable, GL_DEPTH_TEST, glClearColor
            from OpenGL.GLU import gluPerspective
        except Exception:
            self._gl_ok = False
            self.model.mesh_error = 'PyOpenGL not available; install PyOpenGL to enable 3D preview.'
            return

        self._gl_ok = True
        from OpenGL.GL import glEnable, GL_DEPTH_TEST, glClearColor
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.15, 0.15, 0.15, 1.0)

        # Ensure mesh is available (lazy load)
        if not self.model.mesh_loaded and not self.model.mesh_error:
            try:
                verts, norms, uvs, inds = ModelMeshLoader.load_mesh(
                    self.model.mdl_path, self.model.vvd_path, self.model.vtx_path,
                    self.model
                )
                self.model.vertices = verts
                self.model.normals = norms
                self.model.uvs = uvs
                self.model.indices = inds
                self.model.mesh_loaded = True
            except Exception as e:
                self.model.mesh_error = str(e)

    def resizeGL(self, w: int, h: int):
        if not self._gl_ok:
            return
        from OpenGL.GL import glViewport
        glViewport(0, 0, w, h)

    def paintGL(self):
        if not getattr(self, '_gl_ok', False):
            # draw informative text using Qt painter
            self._paint_placeholder()
            return

        from OpenGL.GL import (
            glClear, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
            glMatrixMode, GL_PROJECTION, GL_MODELVIEW, glLoadIdentity,
            glRotatef, glColor3f, glBegin, GL_TRIANGLES, glVertex3f, glEnd
        )
        from OpenGL.GLU import gluPerspective, gluLookAt

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if not self.model.mesh_loaded or not self.model.vertices or not self.model.indices:
            self._paint_placeholder()
            return

        w = max(1, self.width())
        h = max(1, self.height())
        aspect = w / float(h)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, aspect, 0.1, 10000.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # camera
        # compute a basic camera distance based on bounding box
        verts = self.model.vertices
        # compute center
        cx = sum(v[0] for v in verts) / len(verts)
        cy = sum(v[1] for v in verts) / len(verts)
        cz = sum(v[2] for v in verts) / len(verts)
        # simple distance
        max_extent = 0.0
        for v in verts:
            dx = v[0] - cx
            dy = v[1] - cy
            dz = v[2] - cz
            d = (dx*dx + dy*dy + dz*dz) ** 0.5
            if d > max_extent:
                max_extent = d
        dist = max(32.0, max_extent * 3.0)

        gluLookAt(0, 0, dist, 0, 0, 0, 0, 1, 0)
        glRotatef(self.rot_y, 0, 1, 0)

        # simple color
        glColor3f(0.8, 0.8, 0.8)
        glBegin(GL_TRIANGLES)
        for i in self.model.indices:
            x, y, z = self.model.vertices[i]
            glVertex3f(x - cx, y - cy, z - cz)
        glEnd()

    def _paint_placeholder(self):
        # Use Qt painter to show message when GL not available
        from PyQt5.QtGui import QPainter
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        painter.setPen(QColor(200, 200, 200))
        msg = self.model.mesh_error or "No mesh available for preview"
        painter.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap, msg)
        painter.end()

    def mousePressEvent(self, event):
        self._last_x = event.x()

    def mouseMoveEvent(self, event):
        if self._last_x is None:
            self._last_x = event.x()
            return
        dx = event.x() - self._last_x
        self._last_x = event.x()
        self.rot_y += dx * 0.5
        self.update()



# ---------------------------------------------------------------------------
# Detail dialog
# ---------------------------------------------------------------------------

class DetailDialog(QDialog):
    """Shows metadata for a single MDL model."""

    def __init__(self, model: ModelEntry, placeholder_icon: QPixmap, parent=None):
        super().__init__(parent)
        self.model = model
        self.placeholder_icon = placeholder_icon

        self.setWindowTitle(f"Model Details - {model.name}")
        self.resize(600, 400)

        layout = QVBoxLayout(self)

        # Preview widget (OpenGL) if available, otherwise fallback to icon
        self.preview_widget = None
        if QOpenGLWidget is not None:
            try:
                self.preview_widget = ModelPreviewWidget(model, self)
                self.preview_widget.setMinimumHeight(260)
                layout.addWidget(self.preview_widget)
            except Exception:
                self.preview_widget = None

        if self.preview_widget is None:
            # Fallback static icon preview
            self.preview_label = QLabel()
            self.preview_label.setAlignment(Qt.AlignCenter)
            icon_pix = model.icon or placeholder_icon
            icon_pix = icon_pix.scaled(192, 192, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview_label.setPixmap(icon_pix)
            layout.addWidget(self.preview_label)

        # Info labels
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        info_layout.setContentsMargins(0, 0, 0, 0)

        self.path_label = QLabel()
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.meta_label = QLabel()
        self.meta_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.meta_label.setWordWrap(True)

        self.bbox_label = QLabel()
        self.bbox_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.bbox_label.setWordWrap(True)

        # NEW: textures label (populated later)
        self.textures_label = QLabel()
        self.textures_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.textures_label.setWordWrap(True)

        info_layout.addWidget(self.path_label)
        info_layout.addWidget(self.meta_label)
        info_layout.addWidget(self.bbox_label)
        info_layout.addWidget(self.textures_label)
        layout.addWidget(info_widget)

        # Buttons
        btn_widget = QWidget()
        btn_layout = QHBoxLayout(btn_widget)
        btn_layout.addStretch(1)
        self.open_folder_btn = QPushButton("Open Containing Folder")
        self.find_materials_btn = QPushButton("Find Materials")
        self.close_btn = QPushButton("Close")
        btn_layout.addWidget(self.open_folder_btn)
        btn_layout.addWidget(self.find_materials_btn)
        btn_layout.addWidget(self.close_btn)
        layout.addWidget(btn_widget)

        self.open_folder_btn.clicked.connect(self.open_folder)
        self.find_materials_btn.clicked.connect(self.on_find_materials)
        self.close_btn.clicked.connect(self.accept)

        self._populate()

    def _populate(self):
        m = self.model

        # Path
        path_text = f"<b>MDL path:</b> {m.mdl_path}"
        if not m.magic_ok:
            path_text += "<br><b>Note:</b> Header not recognised as Source MDL (magic != IDST)."
        self.path_label.setText(path_text)

        # Metadata
        pieces = []
        if m.version is not None:
            pieces.append(f"Version: {m.version}")
        if m.checksum is not None:
            pieces.append(f"Checksum: {m.checksum}")
        if m.internal_name:
            pieces.append(f"Internal name: {m.internal_name}")
        if m.file_length is not None:
            pieces.append(f"MDL length (header): {m.file_length} bytes")
        if m.file_size is not None:
            pieces.append(f"File size on disk: {m.file_size} bytes")

        if m.numbones is not None:
            pieces.append(f"Bones: {m.numbones}")
        if m.numlocalanim is not None:
            pieces.append(f"Local animations: {m.numlocalanim}")
        if m.numlocalseq is not None:
            pieces.append(f"Sequences: {m.numlocalseq}")
        if m.numtextures is not None:
            pieces.append(f"Textures (names): {m.numtextures}")
        if m.numbodyparts is not None:
            pieces.append(f"Body parts: {m.numbodyparts}")
        
        if m.flags is not None:
            pieces.append(f"Flags: 0x{m.flags:08X}")

        if not pieces:
            pieces.append("No MDL header metadata available.")

        self.meta_label.setText("<br>".join(pieces))

        # Bounding boxes
        bbox_parts = []
        if m.eye_pos:
            bbox_parts.append(
                "Eye position: (%.2f, %.2f, %.2f)" % m.eye_pos
            )
        if m.hull_min and m.hull_max:
            bbox_parts.append(
                "Hull AABB: min (%.2f, %.2f, %.2f), max (%.2f, %.2f, %.2f)"
                % (*m.hull_min, *m.hull_max)
            )
        if m.view_bbmin and m.view_bbmax:
            bbox_parts.append(
                "View AABB: min (%.2f, %.2f, %.2f), max (%.2f, %.2f, %.2f)"
                % (*m.view_bbmin, *m.view_bbmax)
            )
        if not bbox_parts:
            bbox_parts.append("No bounding box information parsed.")

        self.bbox_label.setText("<br>".join(bbox_parts))

        # Textures & CD textures
        tex_lines: List[str] = []
        if m.texture_names:
            tex_lines.append("<b>Texture names (from MDL):</b>")
            for name in m.texture_names:
                tex_lines.append(name)
        else:
            tex_lines.append("<b>Texture names:</b> (none parsed)")
        if m.cdtexture_paths:
            tex_lines.append("<br><b>CD texture search paths:</b>")
            for p in m.cdtexture_paths:
                tex_lines.append(p)
        else:
            tex_lines.append("<br><b>CD texture paths:</b> (none parsed)")

        self.textures_label.setText("<br>".join(tex_lines))

    def on_find_materials(self):
        """Search the current project root for guessed material (.vmt) files.

        Uses `parent().current_root` when available, otherwise prompts.
        """
        # Determine search root
        root = None
        try:
            parent = self.parent()
        except Exception:
            parent = None
        if parent and hasattr(parent, 'current_root') and parent.current_root:
            root = parent.current_root
        if not root:
            QMessageBox.information(self, "Find Materials", "No project root selected. Open a folder first in the main window.")
            return

        m = self.model
        candidates = []
        # If no cdtexture paths, use empty prefix (engine may also use other search paths)
        cd_paths = m.cdtexture_paths or [""]
        tex_names = m.texture_names or []

        for cd in cd_paths:
            # normalize cd path (may contain trailing slashes)
            cd_norm = cd.replace('\\', '/').lstrip('/')
            for tex in tex_names:
                tex_norm = tex.replace('\\', '/').lstrip('/')
                # candidate straightforward path: materials/<cd>/<tex>.vmt
                cand = os.path.normpath(os.path.join(root, 'materials', cd_norm, tex_norm + '.vmt'))
                candidates.append(cand)

        found = []
        # First pass: direct existence check
        for p in candidates:
            if os.path.exists(p):
                found.append(p)

        # Second pass: if nothing found, do a best-effort search under materials/<cd>
        if not found:
            searched = set()
            for cd in cd_paths:
                cd_norm = cd.replace('\\', '/').lstrip('/')
                base_dir = os.path.join(root, 'materials', cd_norm)
                if not os.path.isdir(base_dir) or base_dir in searched:
                    continue
                searched.add(base_dir)
                for dirpath, dirnames, filenames in os.walk(base_dir):
                    for fname in filenames:
                        if not fname.lower().endswith('.vmt'):
                            continue
                        full = os.path.join(dirpath, fname)
                        # If any texture name is a suffix of the vmt path (ignoring extension), consider it a match
                        rel = os.path.relpath(full, os.path.join(root, 'materials')).replace('\\', '/')
                        for tex in tex_names:
                            tex_s = tex.replace('\\', '/').lstrip('/')
                            if rel.lower().endswith((tex_s + '.vmt').lower()):
                                if full not in found:
                                    found.append(full)
                                    break

        # Present results
        if not found:
            QMessageBox.information(self, "Find Materials", "No matching .vmt files found for the model's textures under the selected root.")
            return

        # Show results in a simple dialog with a list
        dlg = QDialog(self)
        dlg.setWindowTitle("Found Materials")
        dlg.resize(700, 400)
        layout = QVBoxLayout(dlg)
        info = QLabel(f"Found {len(found)} .vmt files (double-click to open containing folder):")
        layout.addWidget(info)
        listw = QListWidget()
        for p in found:
            item = QListWidgetItem(p)
            listw.addItem(item)
        layout.addWidget(listw)

        btn_close = QPushButton("Close")
        btn_open_folder = QPushButton("Open Containing Folder")
        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)
        btn_layout.addWidget(btn_open_folder)
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

        def open_selected_folder():
            sel = listw.currentItem()
            if not sel:
                return
            path = sel.text()
            if os.path.exists(path):
                folder = os.path.dirname(path)
                open_in_file_explorer(folder)

        listw.itemDoubleClicked.connect(lambda it: open_in_file_explorer(os.path.dirname(it.text())))
        btn_open_folder.clicked.connect(open_selected_folder)
        btn_close.clicked.connect(dlg.accept)

        dlg.exec_()

    def open_folder(self):
        if not self.model.mdl_path or not os.path.exists(self.model.mdl_path):
            QMessageBox.warning(self, "Open Folder", "File no longer exists on disk.")
            return
        open_in_file_explorer(self.model.mdl_path)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Source MDL Browser (PyQt)")
        self.resize(1200, 700)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Data
        self.current_root: Optional[str] = None
        self.models: Dict[int, ModelEntry] = {}
        self.scan_thread: Optional[ScanThread] = None

        # Filters
        self.current_dir_filter: str = ""  # relative directory; '' = all
        self.search_text: str = ""

        # Directory tree mapping: rel_dir -> QTreeWidgetItem
        self.dir_items: Dict[str, QTreeWidgetItem] = {}

        # Icon
        self.model_icon = create_model_icon(128)

        self._init_menu()
        self._init_ui()

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

        # Right side: filter bar + list
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        filter_widget = QWidget()
        filter_layout = QHBoxLayout(filter_widget)
        filter_layout.setContentsMargins(0, 0, 0, 0)

        filter_label = QLabel("Filter:")
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Search by name or path...")
        self.filter_edit.textChanged.connect(self.on_filter_text_changed)

        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.filter_edit)
        right_layout.addWidget(filter_widget)

        self.thumb_list = QListWidget()
        self.thumb_list.setViewMode(QListWidget.IconMode)
        self.thumb_list.setResizeMode(QListWidget.Adjust)
        self.thumb_list.setMovement(QListWidget.Static)
        self.thumb_list.setIconSize(QSize(128, 128))
        self.thumb_list.setSpacing(8)
        self.thumb_list.setWordWrap(True)

        # Fix per-item cell size to avoid layout glitches when icons/text differ
        self.default_item_size = QSize(150, 180)

        self.thumb_list.itemClicked.connect(self.on_item_clicked)
        self.thumb_list.itemDoubleClicked.connect(self.on_item_double_clicked)

        splitter.addWidget(self.thumb_list)
        splitter.setStretchFactor(1, 1)

    # ---- Folder handling and scanning ----

    def open_folder(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Root Folder Containing MDL Files",
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
        # Stop existing thread
        if self.scan_thread and self.scan_thread.isRunning():
            self.scan_thread.requestInterruption()
            self.scan_thread.wait(1000)

        self.current_root = root_path
        self.models.clear()
        self.thumb_list.clear()
        self.dir_tree.clear()
        self.dir_items.clear()
        self.current_dir_filter = ""
        self.search_text = ""
        # Optional: if you *really* want to clear the filter, wrap safely:
        # if getattr(self, "filter_edit", None) is not None:
        #     try:
        #         self.filter_edit.setText("")
        #     except RuntimeError:
        #         pass

        # Root tree item
        root_name = os.path.basename(root_path.rstrip(os.sep)) or root_path
        root_item = QTreeWidgetItem([root_name])
        root_item.setData(0, Qt.UserRole, "")
        self.dir_tree.addTopLevelItem(root_item)
        self.dir_tree.expandItem(root_item)
        self.dir_items[""] = root_item

        self.status_bar.showMessage(f"Scanning: {root_path} ...")

        self.scan_thread = ScanThread(root_path)
        self.scan_thread.modelFound.connect(self.on_model_found)
        self.scan_thread.progress.connect(self.on_scan_progress)
        self.scan_thread.finishedScanning.connect(self.on_scan_finished)
        self.scan_thread.start()

    def on_scan_progress(self, message: str):
        self.status_bar.showMessage(message)

    def on_model_found(self, entry_obj):
        entry: ModelEntry = entry_obj
        self.models[entry.id] = entry
        self._ensure_directory_item(entry.directory)
        if self._entry_matches_filters(entry):
            self._add_list_item_for_entry(entry)

    def on_scan_finished(self):
        if not self.models:
            self.status_bar.showMessage("Scan complete. No MDL files found.")
            return
        self.status_bar.showMessage(f"Scan complete. Found {len(self.models)} models.")

    # ---- Directory tree ----

    def _ensure_directory_item(self, rel_dir: str):
        if rel_dir in self.dir_items:
            return

        parts = [p for p in rel_dir.replace('\\', '/').split('/') if p]
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

    # ---- Filtering ----

    def on_filter_text_changed(self, text: str):
        self.search_text = text.strip().lower()
        self.refresh_list_from_filters()

    def _entry_matches_filters(self, entry: ModelEntry) -> bool:
        # Directory filter
        if self.current_dir_filter:
            if entry.directory == self.current_dir_filter:
                pass
            elif entry.directory.startswith(self.current_dir_filter + "/"):
                pass
            else:
                return False

        # Text filter
        if self.search_text:
            haystack = [
                entry.name.lower(),
                entry.directory.lower(),
                entry.mdl_path.lower(),
                (entry.internal_name or "").lower(),
            ]
            if not any(self.search_text in s for s in haystack):
                return False

        return True

    def refresh_list_from_filters(self):
        self.thumb_list.clear()
        for entry in self.models.values():
            if self._entry_matches_filters(entry):
                self._add_list_item_for_entry(entry)

    # ---- List item handling ----

    def _add_list_item_for_entry(self, entry: ModelEntry):
        text = entry.name
        if not entry.magic_ok:
            text += "\n(invalid header?)"

        item = QListWidgetItem(text)
        item.setData(Qt.UserRole, entry.id)
        item.setSizeHint(self.default_item_size)

        # Icon: generic model cube
        if entry.icon is None:
            entry.icon = self.model_icon
        item.setIcon(QIcon(entry.icon))

        self.thumb_list.addItem(item)

    def _get_entry_for_item(self, item: QListWidgetItem) -> Optional[ModelEntry]:
        entry_id = item.data(Qt.UserRole)
        return self.models.get(entry_id)

    def on_item_clicked(self, item: QListWidgetItem):
        entry = self._get_entry_for_item(item)
        if not entry:
            return
        dlg = DetailDialog(entry, self.model_icon, self)
        dlg.exec_()

    def on_item_double_clicked(self, item: QListWidgetItem):
        entry = self._get_entry_for_item(item)
        if not entry:
            return
        if not entry.mdl_path or not os.path.exists(entry.mdl_path):
            QMessageBox.warning(self, "Open Folder", "File no longer exists on disk.")
            return
        open_in_file_explorer(entry.mdl_path)

    # ---- Cleanup ----

    def closeEvent(self, event):
        if self.scan_thread and self.scan_thread.isRunning():
            self.scan_thread.requestInterruption()
            self.scan_thread.wait(1000)
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
