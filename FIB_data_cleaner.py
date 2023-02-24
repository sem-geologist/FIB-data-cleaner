import sys
import json
import gc
import numpy as np
from os import path
from copy import copy
from PyQt5 import QtCore, QtGui, QtWidgets
import hyperspy.api as hs
import pyqtgraph as pg
from ui import MainWindow
from scipy.ndimage import affine_transform
pg.setConfigOption('imageAxisOrder', 'row-major')
from affine6p import estimate
from ui.CustomComponents import (ZAlignmentROI,
                                 SpinBoxDelegate,
                                 TransformationMatrixModel)


class FIBSliceCorrector(QtWidgets.QMainWindow,
                        MainWindow.Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)
        self.first_time_load = True
        self.at_matrices = {}  # affine transformation matrices
        delegate = SpinBoxDelegate()
        self.mtv.setItemDelegate(delegate)
        self.actionLoad.triggered.connect(self.load_single_file)
        self.actionAbout_Qt.triggered.connect(self.show_about_qt)
        self.actionAbout_this_software.triggered.connect(self.show_about)
        self.v_depth_iv.ui.roiBtn.hide()
        self.h_depth_iv.ui.roiBtn.hide()
        self.h_depth_iv.getView().invertY(False)
        self.slice_iv.ui.roiBtn.hide()
        self.slice_iv.ui.menuBtn.hide()
        self.h_depth_iv.ui.menuBtn.hide()
        self.v_depth_iv.ui.menuBtn.hide()
        # simple shift (True), affine matrix (False):
        self.simple_mode = True
        self.reset_triangle.setIcon(pg.icons.default.qicon)
        kernel = self.console.kernel_manager.kernel
        kernel.shell.push(dict(np=np, pg=pg, hs=hs, app=self))
        self.splitter_4.addWidget(self.console)
        self.console.execute("whos")
        self.console.setVisible(False)

    @classmethod
    def init_affine_transformation_matrix(cls):
        return np.array([[1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]],
                        dtype=np.float64)

    def get_empty_slice(self):
        if self.i_data is None:
            raise ValueError("no data is loaded")
        else:
            return [slice(None)] * self.i_data.ndim

    def get_frame_slice(self, i_frame):
        i_slice = self.get_empty_slice()
        i_slice[self.i_depth] = i_frame
        return (*i_slice,)

    def get_index_for_manipulation(self):
        if self.slice_lock.isChecked():
            return self.locked_index
        return self.current_i

    def init_at_at_index(self, index_z=None):
        """generate and set initial affine transformation matrix
           for given index, also generate model, which can be
           viewed with QTableView"""
        if index_z is None:
            current_flag = True
            index_z = self.get_index_for_manipulation()
        if index_z not in self.at_matrices:
            self.at_matrices[index_z] = {
                'matrix': self.init_affine_transformation_matrix()}
            TransformationMatrixModel(self.at_matrices[index_z])
            i_slice = self.get_frame_slice(index_z)
            self.at_matrices[index_z]['initial'] = copy(self.i_data[i_slice])
            if current_flag:
                self.switch_to_matrix_correction_widget()

    def remove_at_and_reset(self):
        i = self.get_index_for_manipulation()
        i_slice = self.get_frame_slice(i)
        self.i_data[i_slice] = self.at_matrices[i]['initial']
        self.at_matrices[i]['model'].dataChanged.disconnect(
            self.apply_affine_transformation)
        self.mtv.setModel(None)
        del self.at_matrices[i]
        self.mtv.setEnabled(False)
        self.reset_matrix.setEnabled(False)
        self.update_images()
        self.switch_to_simple_shift_widget()

    def switch_to_matrix_correction_widget(self):
        i = self.get_index_for_manipulation()
        self.mtv.setModel(self.at_matrices[i]['model'])
        self.show_ref_points.setEnabled(True)
        self.reset_matrix.setEnabled(True)
        self.init_slice_btn.setEnabled(False)
        self.mtv.setEnabled(True)
        self.at_matrices[i]['model'].dataChanged.connect(
            self.apply_affine_transformation)
        self.sc_group.setEnabled(False)
        self.to_end_checkbox.setChecked(False)
        self.slices_spinbox.setValue(1)
        self.simple_mode = False

    def switch_to_simple_shift_widget(self):
        if self.show_ref_points.isChecked():
            self.show_ref_points.setChecked(False)
        self.show_ref_points.setEnabled(False)
        self.reset_matrix.setEnabled(False)
        self.init_slice_btn.blockSignals(True)
        self.init_slice_btn.setChecked(False)
        self.init_slice_btn.blockSignals(False)
        self.init_slice_btn.setEnabled(True)
        self.mtv.setEnabled(False)
        self.mtv.setModel(None)
        self.sc_group.setEnabled(True)
        self.simple_mode = True

    def apply_shifts(self, shifts):
        i_slice = self.get_empty_slice()
        for i in range(self.i_data.shape[self.i_depth]):
            i_slice[self.i_depth] = i
            self.i_data[(*i_slice,)] = np.roll(self.i_data[(*i_slice,)],
                                               shifts[i],
                                               axis=(1, 0))
            #self.i_data[(*i_slice,)] = np.roll(self.i_data[(*i_slice,)],
            #                                   shifts[i, 0],
            #                                   axis=1)
            #self.i_data[i_slice] = np.roll(self.i_data[(*i_slice,)],
            #                               shifts[i, 1],
            #                               axis=0)

    def save_corrections(self):
        at_jsonized = {str(i): self.at_matrices[i]['matrix'].tolist()
                       for i in self.at_matrices}
        roi_jsonized = {'pos': tuple(self.roi_main.pos()),
                        'size': tuple(self.roi_main.size())}
        jsonable_dict = {'shifts': self.shifts.tolist(),
                         'at': at_jsonized,
                         'roi': roi_jsonized}
        fn, ft = QtWidgets.QFileDialog.getSaveFileName(
            self, 'give name for correction file', '../',
            "Json correction file (*.json)")
        if ft == '':
            return
        with open(fn, 'w') as fp:
            json.dump(jsonable_dict, fp)

    def load_shifts(self):
        """load and apply pixel shifts"""
        fn, ft = QtWidgets.QFileDialog.getOpenFileName(
            self, 'select correction file', '../',
            "Numpy files (*.npy);; Json correction (*.json)")
        if ft == '':
            return
        self.apply_shifts(-self.shifts)
        if '.npy' in ft:
            shifts = np.load(fn).astype(np.int32)
            self.apply_shifts(shifts)
            self.shifts = shifts
        elif '.json' in ft:
            with open(fn, 'r') as file_pointer:
                corrections = json.load(file_pointer)
            self.shifts = np.asarray(corrections['shifts'], dtype=np.int32)
            self.apply_shifts(self.shifts)
            if 'roi' in corrections:
                self.roi_main.setPos(corrections['roi']['pos'])
                self.roi_main.setSize(corrections['roi']['size'])
            at = {int(i): np.asarray(corrections['at'][i],
                                     dtype=np.float64)
                  for i in corrections['at']}
            i_slice = self.get_empty_slice()
            for i in at:
                self.at_matrices[i] = {'matrix': at[i]}
                TransformationMatrixModel(self.at_matrices[i])
                i_slice[self.i_depth] = i
                self.at_matrices[i]['initial'] = copy(
                    self.i_data[(*i_slice,)])
                self.apply_affine_transformation(i)
        self.update_shift_widget()
        self.which_widget()
        self.slice_iv.updateImage()
        self.v_depth_iv.updateImage()
        self.h_depth_iv.updateImage()

    def connection_set_after_load(self):
        self.left_button.pressed.connect(self.shift_left)
        self.right_button.pressed.connect(self.shift_right)
        self.up_button.pressed.connect(self.shift_up)
        self.down_button.pressed.connect(self.shift_down)
        self.slice_lock.toggled.connect(self.lock_current_index)
        self.actionLoad_corrections.setEnabled(True)
        self.actionLoad_corrections.triggered.connect(self.load_shifts)
        self.actionGet_z_scale.setEnabled(True)
        self.actionGet_z_scale.triggered.connect(self.update_aspect_ratio)
        self.actionshow_markers.toggled.connect(self.visual_guides)
        self.v_slice_line.sigPositionChanged.connect(
                                                self.change_vertical_slice)
        self.h_slice_line.sigPositionChanged.connect(
                                                self.change_horizontal_slice)
        self.init_slice_btn.setEnabled(True)
        self.init_slice_btn.pressed.connect(self.init_at_at_index)
        self.reset_matrix.pressed.connect(self.remove_at_and_reset)
        self.actionSave_corrections.setEnabled(True)
        self.actionSave_corrections.triggered.connect(self.save_corrections)
        self.show_ref_points.toggled.connect(self.show_original_triangulation)
        self.show_ref_points.toggled.connect(self.enable_affine_by_tripoint)
        self.pin_points_to_slice.toggled.connect(self.enable_affine_by_triangle)

    def enable_affine_by_tripoint(self, state):
        if state:
            self.pin_points_to_slice.setEnabled(True)
        else:
            self.pin_points_to_slice.setChecked(False)
            self.pin_points_to_slice.setEnabled(False)

    def visual_guides(self, status):
        self.align_line_x1.setVisible(status)
        self.align_line_x2.setVisible(status)
        self.align_line_y1.setVisible(status)
        self.align_line_y2.setVisible(status)
        self.roi_main.setVisible(status)

    def change_vertical_slice(self):
        pos = int(round(self.v_slice_line.pos()[0]))
        i_slice = self.get_empty_slice()
        i_slice[self.i_width] = pos
        self.slice_yz = self.i_data[(*i_slice,)]
        if self.i_depth < self.i_height:
            i_z, i_y = 0, 1
        else:
            i_z, i_y = 1, 0
        self.v_depth_iv.setImage(self.slice_yz,
                                 axes={'x': i_z, 'y': i_y},
                                 autoRange=False,
                                 autoHistogramRange=False,
                                 autoLevels=False)

    def change_horizontal_slice(self):
        pos = int(round(self.h_slice_line.pos()[1]))
        i_slice = self.get_empty_slice()
        i_slice[self.i_height] = pos
        self.slice_xz = self.i_data[(*i_slice,)]
        if self.i_depth < self.i_width:
            i_z, i_x = 0, 1
        else:
            i_z, i_x = 1, 0
        self.h_depth_iv.setImage(self.slice_xz,
                                 axes={'x': i_x, 'y': i_z},
                                 autoRange=False,
                                 autoHistogramRange=False,
                                 autoLevels=False)

    def lock_current_index(self, state):
        if state:
            self.locked_index = self.current_i
            i_slice = self.get_frame_slice(self.locked_index)
            self.locked_image_item = pg.ImageItem(self.i_data[i_slice])
            self.slice_iv.addItem(self.locked_image_item)
            self.slice_lock.setText(
                ' locked at index {}'.format(self.locked_index))
            self.transparency_slider.sliderMoved.connect(
                                        self.change_locked_image_opacity)
            self.change_locked_image_opacity(
                self.transparency_slider.value())
            self.horiz_line_2.hide()
            self.vert_line_2.hide()
            self.horiz_line_locked.setPos(self.locked_index)
            self.horiz_line_locked.show()
            self.vert_line_locked.setPos(self.locked_index)
            self.vert_line_locked.show()
        else:
            self.slice_lock.setText(
                  ' Lock onto current slice:')
            self.slice_iv.removeItem(self.locked_image_item)
            self.locked_image_item = None
            self.horiz_line_2.show()
            self.vert_line_2.show()
            self.horiz_line_locked.hide()
            self.vert_line_locked.hide()
            self.update_shift_widget()
            self.which_widget()

    def change_locked_image_opacity(self, int_value):
        self.locked_image_item.setOpacity(int_value / 100)

    def enable_affine_by_triangle(self, state):
        for i in self.original_ref:
            i.movable = not state
        if state:
            points = [(j.x(), j.y()) for j in self.original_ref]
            self.floating_triangle = pg.PolyLineROI(positions=points,
                                                    closed=True)
            self.slice_iv.addItem(self.floating_triangle)
            self.floating_triangle.sigRegionChanged.connect(self.at_from_three)
        else:
            self.floating_triangle.sigRegionChanged.disconnect(
                self.at_from_three)
            self.slice_iv.removeItem(self.floating_triangle)
            self.reset_triangle.setDisabled(True)

    def at_from_three(self):
        index = self.get_index_for_manipulation()
        h_list = [tuple(h.pos()) for h in self.floating_triangle.getHandles()]
        p_list = [tuple(p.pos()) for p in self.original_ref]
        self.at_matrices[index]['matrix'][:] = estimate(h_list, p_list).get_matrix()
        self.apply_affine_transformation(index)
        model = self.at_matrices[index]["model"]
        model.dataChanged.emit(model.index(0, 0), model.index(3, 3))

    def save_cube(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName()
        if fn is None:
            return
        self.cube.save(fn)

    def crop_cube(self):
        if self.actionlock_onto_current_slice.isChecked():
            self.actionlock_onto_current_slice.setChecked(False)
        self.cube.crop(self.i_height, start=int(self.align_line_x1.pos()[0]),
                       end=int(self.align_line_x2.pos()[0]))
        self.cube.crop(self.i_width, start=int(self.align_line_y1.pos()[1]),
                       end=int(self.align_line_y2.pos()[1]))
        self.i_data = self.cube.data
        self.shifts = np.zeros(shape=(self.i_data.shape[self.i_depth], 2),
                               dtype=np.int32)
        self.at_matrices = {}
        self.switch_to_simple_shift_widget()
        self.align_line_x1.setPos(0)
        self.align_line_x2.setPos(self.i_data.shape[self.i_width])
        self.align_line_y1.setPos(0)
        self.align_line_y2.setPos(self.i_data.shape[self.i_height])
        self.slice_iv.setImage(self.i_data, axes={"t": self.i_depth,
                                                  "x": self.i_width,
                                                  "y": self.i_height})
        self.setup_slicers()

    def normalize(self):
        x1 = int(self.align_line_x1.pos()[0])
        x2 = int(self.align_line_x2.pos()[0])
        y1 = int(self.align_line_y1.pos()[1])
        y2 = int(self.align_line_y2.pos()[1])

        i_slice = self.get_empty_slice()
        i_slice[self.i_width] = slice(x1, x2)
        i_slice[self.i_height] = slice(y1, y2)
        norma_arr = np.mean(self.i_data[(*i_slice,)],
                            axis=(self.i_height, self.i_width))
        min_arr = int(norma_arr.min())
        normalization_arr = norma_arr - min_arr
        i_slice = self.get_empty_slice()
        for i in range(self.i_data.shape[self.i_depth]):
            i_slice = self.get_frame_slice(i)
            self.i_data[i_slice] = np.where(
                self.i_data[i_slice] < int(normalization_arr[i] - 1),
                self.i_data[i_slice],
                self.i_data[i_slice] - int(normalization_arr[i] - 1))
        self.slice_iv.updateImage()
        self.v_depth_iv.updateImage()
        self.h_depth_iv.updateImage()

    def load_single_file(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName()
        if fn is None or fn == "":
            return
        self.setCursor(QtCore.Qt.WaitCursor)
        self.unload_file()
        signal = hs.load(fn)
        if type(signal) == list:
            str_signals = [str(i) for i in signal]
            selection = QtWidgets.QInputDialog.getItem(
                self,
                "Signal",
                "Few signals were found inside the file;\n"
                "please, choose one to proceed further", 
                str_signals)
            if selection[1]:
                index = str_signals.index(selection[0])
                signal = signal[index]
            else:
                self.unsetCursor()
                return
        self.load_hspy_signal(signal)
        self.unsetCursor()

    def unload_file(self):
        if self.actionlock_onto_current_slice.isChecked():
            self.actionlock_onto_current_slice.setChecked(False)
        self.cube = None
        self.i_data = None
        self.shifts = None
        self.slice_iv.clear()
        self.slice_xz = None
        self.slice_yz = None
        self.v_depth_iv.clear()
        self.h_depth_iv.clear()
        self.at_matrices = {}  # reset transformation matrices
        self.actionVertical_shift_guide.setChecked(False)
        self.actionHorizontal_shift_guide.setChecked(False)
        gc.collect()

    def set_index_by_name(self, hspy_sig):
        hd = hspy_sig.axes_manager.as_dictionary()
        n_axes = len(hd)
        for i in range(n_axes):
            if hspy_sig.axes_manager[i].name == 'width':
                self.i_width = hspy_sig.axes_manager[i].index_in_array
            elif hspy_sig.axes_manager[i].name == 'height':
                self.i_height = hspy_sig.axes_manager[i].index_in_array
            elif n_axes == 3:
                self.i_depth = hspy_sig.axes_manager[i].index_in_array
            else:
                raise TypeError("data with more than 3 dimensions currently not supported")

    def load_hspy_signal(self, signal):
        self.unload_file()
        self.setWindowTitle(signal.metadata.General.original_filename)
        self.cube = signal
        self.set_index_by_name(signal)
        self.i_data = self.cube.data
        self.shifts = np.zeros(shape=(self.i_data.shape[self.i_depth], 2))
        self.shifts = self.shifts.astype(np.int32)
        self.slice_iv.setImage(self.i_data, axes={'t': self.i_depth,
                                                  'x': self.i_width,
                                                  'y': self.i_height})
        self.current_i = self.slice_iv.currentIndex
        self.current_slice_n.setText(str(self.current_i))
        self.main_image_item = self.slice_iv.getImageItem()
        self.main_image_item.sigImageChanged.connect(self.update_index)
        self.setup_slicers()
        if self.first_time_load:
            self.gen_align_lines()
            self.connection_set_after_load()
            self.actionSave.setEnabled(True)
            self.actionSave.triggered.connect(self.save_cube)
            self.actionCrop.setEnabled(True)
            self.actionCrop.triggered.connect(self.crop_cube)
            self.actionNormalize.triggered.connect(self.normalize)
            self.actionVertical_shift_guide.toggled.connect(self.graphic_v_guide)
            self.actionHorizontal_shift_guide.toggled.connect(self.graphic_h_guide)
            self.slices_spinbox.valueChanged.connect(
                self.update_index)
            self.v_roi_line_btn = QtWidgets.QPushButton("aligning to Polyline ROI")
            self.v_depth_iv.ui.gridLayout.addWidget(self.v_roi_line_btn)
            self.v_roi_line_btn.setVisible(False)
            self.v_roi_line_btn.pressed.connect(self.align_to_v_guide_roi)
            self.h_roi_line_btn = QtWidgets.QPushButton("aligning to Polyline ROI")
            self.h_depth_iv.ui.gridLayout.addWidget(self.h_roi_line_btn)
            self.h_roi_line_btn.setVisible(False)
            self.h_roi_line_btn.pressed.connect(self.align_to_h_guide_roi)
            self.first_time_load = False
        
    def graphic_v_guide(self, state):
        self.v_roi_line_btn.setVisible(state)
        if state:
            self.v_guide_roi = ZAlignmentROI(self, orientation="h")
            self.v_depth_iv.addItem(self.v_guide_roi)
        else:
            self.v_depth_iv.removeItem(self.v_guide_roi)
            self.v_guide_roi = None

    def align_to_v_guide_roi(self):
        shifts = self.v_guide_roi.getShifts()
        self.actionVertical_shift_guide.setChecked(False)
        self.apply_shifts(shifts)
        self.shifts += shifts
        self.update_images()

    def graphic_h_guide(self, state):
        self.h_roi_line_btn.setVisible(state)
        if state:
            self.h_guide_roi = ZAlignmentROI(self, orientation="v")
            self.h_depth_iv.addItem(self.h_guide_roi)
        else:
            self.h_depth_iv.removeItem(self.h_guide_roi)
            self.h_guide_roi = None

    def align_to_h_guide_roi(self):
        shifts = self.h_guide_roi.getShifts()
        self.actionHorizontal_shift_guide.setChecked(False)
        self.apply_shifts(shifts)
        self.shifts += shifts
        self.update_images()

    def setup_slicers(self):
        v_bound = self.i_data.shape[self.i_width]
        h_bound = self.i_data.shape[self.i_height]
        if self.first_time_load:
            self.horiz_line = pg.InfiniteLine(pos=0, angle=0,
                                              movable=False, pen='g')
            self.h_depth_iv.addItem(self.horiz_line)
            self.horiz_line_2 = pg.InfiniteLine(
                pos=1, angle=0, movable=False,
                pen=pg.mkPen('g', style=QtCore.Qt.DotLine))
            self.horiz_line_locked = pg.InfiniteLine(
                pos=0, angle=0, movable=False, pen='r')
            self.h_depth_iv.addItem(self.horiz_line_locked)
            self.horiz_line_locked.hide()
            self.h_depth_iv.addItem(self.horiz_line_2)
            self.vert_line = pg.InfiniteLine(pos=0, angle=90, movable=False,
                                             pen='g')
            self.vert_line_2 = pg.InfiniteLine(
                pos=1, angle=90, movable=False,
                pen=pg.mkPen('g', style=QtCore.Qt.DotLine))
            self.vert_line_locked = pg.InfiniteLine(
                pos=0, angle=90, movable=False, pen='r')
            self.v_depth_iv.addItem(self.vert_line_locked)
            self.vert_line_locked.hide()
            self.v_depth_iv.addItem(self.vert_line)
            self.v_depth_iv.addItem(self.vert_line_2)
            self.v_slice_line = pg.InfiniteLine(
                pos=v_bound // 2, movable=True,
                pen='b', bounds=[0, v_bound - 1])
            self.h_slice_line = pg.InfiniteLine(
                pos=h_bound // 2, angle=0,
                movable=True, pen='b',
                bounds=[0, h_bound - 1])
            self.slice_iv.addItem(self.v_slice_line)
            self.slice_iv.addItem(self.h_slice_line)
        else:
            self.v_slice_line.setPos(v_bound // 2)
            self.v_slice_line.setBounds([0, v_bound - 1])
            self.h_slice_line.setPos(h_bound // 2)
            self.h_slice_line.setBounds([0, h_bound - 1])
        xz_slice = self.get_empty_slice()
        xz_slice[self.i_height] = h_bound // 2
        self.slice_xz = self.i_data[(*xz_slice,)]
        yz_slice = self.get_empty_slice()
        yz_slice[self.i_width] = v_bound // 2
        self.slice_yz = self.i_data[(*yz_slice,)]
        if self.i_depth < self.i_height:
            i_z, i_y = 0, 1
        else:
            i_z, i_y = 1, 0
        self.v_depth_iv.setImage(self.slice_yz,
                                 axes={'x': i_z, 'y': i_y})
        if self.i_depth < self.i_width:
            i_z, i_x = 0, 1
        else:
            i_z, i_x = 1, 0
        self.h_depth_iv.setImage(self.slice_xz,
                                 axes={'x': i_x, 'y': i_z})
        self.gen_original_triangulation_ref()

    def gen_original_triangulation_ref(self):
        x1 = self.i_data.shape[self.i_width] // 10
        x2 = x1 * 9
        y1 = self.i_data.shape[self.i_height] // 10
        y2 = y1 * 9
        self.original_ref = [pg.TargetItem(pos=(x1, y1)),
                             pg.TargetItem(pos=(x2, y1)),
                             pg.TargetItem(pos=(x2, y2))]

    def show_original_triangulation(self, state):
        if state:
            for i in self.original_ref:
                self.slice_iv.addItem(i)
        else:
            for i in self.original_ref:
                self.slice_iv.removeItem(i)

    def gen_align_lines(self):
        self.roi_main = pg.RectROI(pos=(20, 20), size=(980, 480))
        self.roi_main.setPen(pg.mkPen('#ffaa00'))
        self.slice_iv.addItem(self.roi_main)
        self.roi_main.hide()
        self.align_line_x1 = pg.InfiniteLine(angle=90, pos=20,
                                             movable=True, pen='y')
        self.align_line_x2 = pg.InfiniteLine(angle=90, pos=1000,
                                             movable=True, pen='y')
        self.h_depth_iv.addItem(self.align_line_x1)
        self.h_depth_iv.addItem(self.align_line_x2)
        self.align_line_y1 = pg.InfiniteLine(angle=0, pos=20,
                                             movable=True, pen='y')
        self.align_line_y2 = pg.InfiniteLine(angle=0, pos=500,
                                             movable=True, pen='y')
        self.v_depth_iv.addItem(self.align_line_y1)
        self.v_depth_iv.addItem(self.align_line_y2)
        self.visual_guides(False)
        self.actionshow_markers.setEnabled(True)
        self.actionshow_markers.setChecked(False)
        for i in ['x1', 'x2', 'y1', 'y2']:
            getattr(self,
                    'align_line_{}'.format(i)).sigPositionChanged.connect(
                                                            self.update_roi)
        self.roi_main.sigRegionChanged.connect(self.update_markers)

    def update_markers(self, roi):
        for i in ['x1', 'x2', 'y1', 'y2']:
            getattr(self,
                    'align_line_{}'.format(i)).sigPositionChanged.disconnect(
                                                            self.update_roi)
        x1, y1 = roi.pos()
        x2, y2 = roi.pos() + roi.size()
        self.align_line_x1.setPos(x1)
        self.align_line_x2.setPos(x2)
        self.align_line_y1.setPos(y1)
        self.align_line_y2.setPos(y2)
        for i in ['x1', 'x2', 'y1', 'y2']:
            getattr(self,
                    'align_line_{}'.format(i)).sigPositionChanged.connect(
                                                             self.update_roi)

    def update_roi(self):
        self.roi_main.sigRegionChanged.disconnect(self.update_markers)
        self.roi_main.setPos((self.align_line_x1.pos()[0],
                              self.align_line_y1.pos()[1]))
        self.roi_main.setSize(
            (self.align_line_x2.pos()[0] - self.align_line_x1.pos()[0],
             self.align_line_y2.pos()[1] - self.align_line_y1.pos()[1]))
        self.roi_main.sigRegionChanged.connect(self.update_markers)

    def update_index(self):
        # print('update index')
        index = self.current_i = self.slice_iv.currentIndex
        strenght = self.slices_spinbox.value()
        self.current_slice_n.setText(str(index))
        self.horiz_line.setPos(index)
        self.horiz_line_2.setPos(index + strenght)
        self.vert_line.setPos(index)
        self.vert_line_2.setPos(index + strenght)
        self.slices_spinbox.setMaximum(
            self.i_data.shape[self.i_depth] - index)
        if not self.slice_lock.isChecked():
            self.update_shift_widget()
            self.which_widget()

    def update_shift_widget(self):
        index = self.get_index_for_manipulation()
        self.label_x_correction.setText(f"x: {self.shifts[index][0]}")
        self.label_y_correction.setText(f"y: {self.shifts[index][1]}")

    def which_widget(self):
        index = self.get_index_for_manipulation()
        if self.simple_mode:
            if index in self.at_matrices:
                self.switch_to_matrix_correction_widget()
        else:
            if index in self.at_matrices:
                self.update_affine_widget()
            else:
                self.switch_to_simple_shift_widget()


    def update_affine_widget(self):
        index = self.get_index_for_manipulation()
        self.mtv.setModel(self.at_matrices[index]["model"])

    def update_aspect_ratio(self):
        init_val = self.cube.axes_manager[0].scale
        z_val, ok_clicked = QtWidgets.QInputDialog.getDouble(
            self, 
            "set interval size", "set inter-slice length \n(changes the aspect ratio for (z,y) and (x,z) plots) \n the distance in-between slices (in nm):",
            init_val,
            0, 500,
            10)
        if ok_clicked:
            self.cube.axes_manager[0].units = 'nm'
            self.cube.axes_manager[0].scale = z_val
            z_val = self.cube.axes_manager[0].scale_as_quantity
            x_val = self.cube.axes_manager[1].scale_as_quantity
            self.v_depth_iv.getView().setAspectLocked(True, ratio=float(z_val/x_val))
            self.h_depth_iv.getView().setAspectLocked(True, ratio=float(x_val/z_val))

    def update_images(self):
        self.slice_iv.updateImage()
        self.v_depth_iv.updateImage()
        self.h_depth_iv.updateImage()
        if self.slice_lock.isChecked():
            self.locked_image_item.updateImage()

    def apply_affine_transformation(self, index=None):
        """scipy.ndimage.affine_transform"""
        if not isinstance(index, int):
            index = self.get_index_for_manipulation()
        i_slice = self.get_frame_slice(index)
        self.i_data[i_slice] = affine_transform(
            self.at_matrices[index]['initial'].T,
            self.at_matrices[index]['matrix']).T
        self.update_images()

    def shift_multi(self, strenght, axis):
        if self.slice_lock.isChecked():
            index = self.locked_index
            end = self.locked_index + 1
        else:
            index = self.current_i
            if self.to_end_checkbox.isChecked():
                end = self.i_data.shape[self.i_depth]
            else:
                end = index + self.slices_spinbox.value()
        for i in range(index, end):
            self.i_data[self.get_frame_slice(i)] = np.roll(
                self.i_data[self.get_frame_slice(i)], strenght, axis=axis)
        if axis == 1:
            self.shifts[index:end, 0] += strenght
        elif axis == 0:
            self.shifts[index:end, 1] += strenght
        self.update_images()
        self.update_shift_widget()

    def shift_right(self):
        strenght = self.strenght_spinbox.value()
        self.shift_multi(strenght, 1)

    def shift_left(self):
        strenght = self.strenght_spinbox.value()
        self.shift_multi(-strenght, 1)

    def shift_up(self):
        strenght = self.strenght_spinbox.value()
        self.shift_multi(-strenght, 0)

    def shift_down(self):
        strenght = self.strenght_spinbox.value()
        self.shift_multi(strenght, 0)

    def show_about_qt(self):
        QtWidgets.QMessageBox.aboutQt(self)

    def show_about(self):
        QtWidgets.QMessageBox.about(self, "About this software",
                                    "Created by Petras Jokubauskas.\n"
                                    "This is free open source software "
                                    "licensed under GPLv3.\n\n"
                                    "Library versions:\n"
                                    f" PyQt5: {QtCore.PYQT_VERSION_STR}\n"
                                    f" PyQtGraph: {pg.__version__}\n"
                                    f" HypersPy: {hs.__version__}\n")


def main():
    app = QtWidgets.QApplication(sys.argv)
    main_window = FIBSliceCorrector()
    main_window.show()
    app.exec()


if __name__ == "__main__":
    main()
