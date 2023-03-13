from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph.graphicsItems.ROI import Handle, PolyLineROI
from pyqtgraph import Point
import numpy as np


class TransformationMatrixModel(QtCore.QAbstractTableModel):
    def __init__(self, dict_node):
        QtCore.QAbstractTableModel.__init__(self)
        dict_node['model'] = self
        self.data_view = dict_node['matrix'].view()

    def rowCount(self, *args):
        return 3

    def columnCount(self, *args):
        return 3

    def headerData(self, section, orientation, role):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                if section == 0:
                    return '(x)'
                if section == 1:
                    return '(y)'
                if section == 2:
                    return 't'
            if orientation == QtCore.Qt.Vertical:
                if section == 0:
                    return 'W'
                elif section == 1:
                    return 'H'
                elif section == 2:
                    return '(z)'

    def data(self, index, role):
        if not index.isValid():
            return False
        if role == QtCore.Qt.DisplayRole or role == QtCore.Qt.EditRole:
            return str(self.data_view[index.row()][index.column()])

        if role == QtCore.Qt.ToolTipRole:
            if index.column() == 2:
                if index.row() == 0:
                    return 'x'
                elif index.row() == 1:
                    return 'y'
            elif index.column() == 0:
                if index.row() == 0:
                    return 'width scaling/rotation(cos(theta))'
                elif index.row() == 1:
                    return 'rotation -sin(theta)/skew in y direction'
            elif index.column() == 1:
                if index.row() == 0:
                    return 'rotation sin(theta)/skew in x direction'
                elif index.row() == 1:
                    return 'height scaling/rotation(cos(theta))'
        if role == QtCore.Qt.BackgroundColorRole:
            if index.row() == index.column():
                return QtGui.QColor(23, 147, 100)

    def flags(self, index):
        if index.isValid():
            if index.row() == 2:
                return QtCore.Qt.ItemIsEnabled
            return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsEditable

    def setData(self, index, value, role):
        if not index.isValid():
            return False
        if role == QtCore.Qt.EditRole:
            self.data_view[index.row()][index.column()] = float(value)
            self.dataChanged.emit(index, index)
            return True


class SpinBoxDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, parent=None):
        QtWidgets.QStyledItemDelegate.__init__(self, parent)

    def createEditor(self, parent, option, index):
        editor = QtWidgets.QDoubleSpinBox(parent)
        editor.setFrame(False)
        if index.column() < 2:
            editor.setMinimum(-10)
            editor.setMaximum(10)
            editor.setSingleStep(0.001)
            editor.setDecimals(3)
        else:
            editor.setMinimum(-1000)
            editor.setMaximum(1000)
            editor.setSingleStep(0.1)
            editor.setDecimals(2)
        return editor

    def setEditorData(self, spinBox, index):
        value = index.model().data(index, QtCore.Qt.EditRole)
        spinBox.setValue(float(value))

    def setModelData(self, editor, model, index):
        value = editor.value()
        model.setData(index, value, QtCore.Qt.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)


class ZAlignmentROI(PolyLineROI):
    def __init__(self, app, orientation="h"):
        self.orientation = orientation
        self.data_cube = app.i_data
        if orientation == "h":
            self.i_l_axis = app.i_depth  # lenght axis index (Z axis)
            self.i_s_axis = app.i_height  # shift axis index
        elif orientation == "v":
            self.i_l_axis = app.i_depth
            self.i_s_axis = app.i_width
        else:
            raise ValueError(
                "orientation kwrd accepts either 'h' or 'v'")
        # initial endpoint_position:
        l_len = self.data_cube.shape[self.i_l_axis] - 1
        s_len = self.data_cube.shape[self.i_s_axis] - 1
        self.l_len = l_len
        self.s_len = s_len
        s_mid = s_len // 5
        if orientation == "h":
            pos1, pos2 = (0, s_mid), (l_len, s_mid)
            max_bounds = QtCore.QRectF(0, 0, l_len, s_len)
        else:
            pos1, pos2 = (s_mid, 0), (s_mid, l_len)
            max_bounds = QtCore.QRectF(0, 0, s_len, l_len)
        PolyLineROI.__init__(self, (pos1, pos2),
                             maxBounds=max_bounds, movable=False,
                             rotatable=False, resizable=False)
        handle_s, handle_e = self.getHandles()
        handle_s.setDeletable(False)
        handle_e.setDeletable(False)

    def movePoint(self, handle, pos, modifiers=None,
                  finish=True, coords='scene'):
        """overwritten method"""
        if modifiers is None:
            modifiers = QtCore.Qt.KeyboardModifier.NoModifier
        index = self.indexOfHandle(handle)
        h = self.handles[index]
        p1 = Point(pos)
        
        #elif coords == 'scene': in this case ALWAYS
        p1 = self.mapSceneToParent(p1) 
        # pixel snap implementation
        p1.setX(round(p1.x()))
        p1.setY(round(p1.y()))
        if self.orientation == 'h':
            if index == 0:
                p1.setX(0)
            elif index == len(self.handles) - 1:
                p1.setX(self.l_len)
            else:
                x_left = self.handles[index - 1]['item'].x()
                x_right = self.handles[index + 1]['item'].x()
                if p1.x() <= x_left:
                    p1.setX(x_left + 1)
                elif p1.x() >= x_right:
                    p1.setX(x_right - 1)
            if p1.y() < 0:
                p1.setY(0)
            elif p1.y() > self.s_len:
                p1.setY(self.s_len)

        else:
            if index == 0:
                p1.setY(0)
            elif index == len(self.handles) - 1:
                p1.setY(self.l_len)
            else:
                y_bottom = self.handles[index - 1]['item'].y()
                y_top = self.handles[index + 1]['item'].y()
                if p1.y() <= y_bottom:
                    p1.setY(y_bottom + 1)
                elif p1.y() >= y_top:
                    p1.setY(y_top - 1)
            if p1.x() < 0:
                p1.setX(0)
            elif p1.x() > self.s_len:
                p1.setX(self.s_len)
        # pushing the constrained values by pos modification
        p_feedback = self.mapToDevice(p1)
        pos.setX(p_feedback.x())
        pos.setY(p_feedback.y())
        ###################################################
        newPos = self.mapFromParent(p1)
        h['item'].setPos(newPos)
        h['pos'] = newPos
        self.freeHandleMoved = True

        self.stateChanged(finish=finish)

    def segmentClicked(self, segment, ev=None, pos=None):
        """overwritten method to prevent inserting of new handle when handles are closer than 2 pixels"""
        if ev is not None:
            pos = segment.mapToParent(ev.pos())
        elif pos is None:
            raise Exception("Either an event or a position must be given.")
        h1 = segment.handles[0]['item']
        h2 = segment.handles[1]['item']
        if self.orientation == 'h':
            if h2.x() - h1.x() < 2:
                return
        elif h2.y() - h1.y() < 2:
            return

        i = self.segments.index(segment)
        h3 = self.addFreeHandle(pos, index=self.indexOfHandle(h2))
        self.addSegment(h3, h2, index=i+1)
        segment.replaceHandle(h2, h3)

    def getShifts(self):
        """return array of frame x, y shifts generated from this Poly
        line ROI"""
        points = np.array([list(Point(i['pos'])) for i in self.handles],
                          dtype=np.int32)
        shifts = np.zeros((self.data_cube.shape[self.i_l_axis], 2))
        if self.orientation == 'h':
            idx = 1
            _idx = 0
        else:
            idx = 0
            _idx = 1
        shifts[:, idx] *= np.nan
        ref_shift = int(points[:, idx].mean())
        points[:, idx] -= ref_shift
        #fill endpoint indexes:
        for p in points:
            shifts[p[_idx], idx] = p[idx]
        # fill nans with interpolation
        target = shifts[:, idx]
        nans, x = nan_helper(target)
        target[nans] = np.interp(x(nans), x(~nans), target[~nans])

        shifts = -np.round(shifts).astype(np.int32)
        return shifts
            

def nan_helper(i):
    return np.isnan(i), lambda z: z.nonzero()[0]

