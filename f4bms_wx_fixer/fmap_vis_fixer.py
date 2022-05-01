import numpy as np
from .util.fileio import FileReader, FileWriter
import os
import shutil
import argparse
import glob
import numpy as np

FMAP_FILE_SIZE = 389916


class FMap(object):
    def __init__(self, fname):
        with FileReader(fname) as reader:
            self.ver = reader.read_int32()
            self.width = reader.read_int32()
            self.height = reader.read_int32()
            self.move_dir = reader.read_int32()
            self.move_vel = reader.read_f32()
            self.unk02 = reader.read_int32()
            self.unk03 = reader.read_int32()
            self.con_alt_sunny_ft = reader.read_int32()
            self.con_alt_fair_ft = reader.read_int32()
            self.con_alt_poor_ft = reader.read_int32()
            self.con_alt_inclement_ft = reader.read_int32()
            total = self.width * self.height

            def read_np(dtype):
                return reader.read_np(dtype,
                                      total).reshape(self.width, self.height)

            self.cloudmap = read_np(np.int32)
            self.pres_mb = read_np(np.float32)
            self.temp_deg_c = read_np(np.float32)
            self.wind_mag_kt = reader.read_np(np.float32, total * 10)
            self.wind_mag_kt = self.wind_mag_kt.reshape(59, 59, 10).T
            self.wind_dir_deg = reader.read_np(np.float32, total * 10)
            self.wind_dir_deg = self.wind_dir_deg.reshape(59, 59, 10).T

            self.cloud_bases_ft = read_np(np.float32)
            self.cloud_coverage = read_np(np.int32)
            self.cloud_size = read_np(np.float32)
            self.tcu = read_np(np.int32)
            self.visibility = read_np(np.float32)

    def save(self, filename):
        with FileWriter(filename) as writer:
            writer.write_int32(self.ver)
            writer.write_int32(self.width)
            writer.write_int32(self.height)
            writer.write_int32(self.move_dir)
            writer.write_f32(self.move_vel)
            writer.write_int32(self.unk02)
            writer.write_int32(self.unk03)
            writer.write_int32(self.con_alt_sunny_ft)
            writer.write_int32(self.con_alt_fair_ft)
            writer.write_int32(self.con_alt_poor_ft)
            writer.write_int32(self.con_alt_inclement_ft)
            writer.write_np(self.cloudmap)
            writer.write_np(self.pres_mb)
            writer.write_np(self.temp_deg_c)
            writer.write_np(self.wind_mag_kt.T)
            writer.write_np(self.wind_dir_deg.T)
            writer.write_np(self.cloud_bases_ft)
            writer.write_np(self.cloud_coverage)
            writer.write_np(self.cloud_size)
            writer.write_np(self.tcu)
            writer.write_np(self.visibility)


def set_min_vis_for_wx_type(fmap, wx_type, min_vis):
    min_vis = float(min_vis)
    fmap.visibility = np.where(
        np.logical_and(fmap.visibility < min_vis, fmap.cloudmap == wx_type),
        min_vis, fmap.visibility)


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(
        description='Fix crazy-low visibility numbers in F4wx fmap outputs')
    parser.add_argument('--input',
                        '-i',
                        required=True,
                        help='Directory containing input fmap files')
    parser.add_argument('--output',
                        '-o',
                        required=True,
                        help='Directory to write output fmap files')
    parser.add_argument('--sunny',
                        '-s',
                        type=float,
                        default=60.0,
                        help='Minimum visibility for sunny cells')
    parser.add_argument('--fair',
                        '-f',
                        type=float,
                        default=40.0,
                        help='Minimum visibility for fair-weather cells')
    parser.add_argument('--poor',
                        '-p',
                        type=float,
                        default=30.0,
                        help='Minimum visibility for poor-weather cells')
    parser.add_argument('--inclement',
                        '-c',
                        type=float,
                        default=20.0,
                        help='Minimum visibility for inclement weather')

    # let the user choose the minimum weather cell level for TCU
    parser.add_argument('--mintcu',
                        '-m',
                        choices=['sunny', 'fair', 'poor', 'inclement'],
                        default='fair',
                        help='Only allow TCU in cells of this type or worse.')
    parse_results = parser.parse_args()

    if parse_results.mintcu is not None:
        mintcu = {
            'sunny': 1,
            'fair': 2,
            'poor': 3,
            'inclement': 4
        }[parse_results.mintcu]
    else:
        mintcu = None

    input_fmaps = [
        f for f in glob.glob(os.path.join(parse_results.input, '*'))
        if os.path.isfile(f) and os.path.getsize(f) == FMAP_FILE_SIZE
    ]

    if len(input_fmaps) == 0:
        raise ValueError(f"No input fmaps found in {parse_results.input}")
    else:
        fmapstr = 'fmap' if len(input_fmaps) == 1 else 'fmaps'
        print(
            f"Found {len(input_fmaps)} {fmapstr} in {os.path.abspath(parse_results.input)}."
        )

    if not os.path.isdir(parse_results.output):
        shutil.os.makedirs(parse_results.output)

    for file_name in input_fmaps:
        fmap = FMap(file_name)
        set_min_vis_for_wx_type(fmap, 1, parse_results.sunny)
        set_min_vis_for_wx_type(fmap, 2, parse_results.fair)
        set_min_vis_for_wx_type(fmap, 3, parse_results.poor)
        set_min_vis_for_wx_type(fmap, 4, parse_results.inclement)
        if mintcu is not None:
            fmap.tcu = np.where(fmap.cloudmap <= mintcu, 0, fmap.tcu)
        fname = os.path.basename(file_name)
        out_file = os.path.join(parse_results.output, fname)
        fmap.save(out_file)
