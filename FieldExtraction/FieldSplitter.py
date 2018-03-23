from FieldExtraction.CoordinateFileReader import CoordinateFileReader
from FieldExtraction.ProgressFileHandler import ProgressFileHandler
from FieldExtraction.ImageParser import ImageParser
from FieldExtraction.ParallelExecutor import ParallelExecutor
from Database.dbHandler import DbHandler


def run(args):
    print(args)
    with CoordinateFileReader(args.coordinate_file, args.img_path_mod, args.image_range) as cf_reader:
        with ProgressFileHandler(args.progress_file) as pf_handler:
            with DbHandler(args.db) as db:
                img_parser = ImageParser(args)
                executor = ParallelExecutor(db, cf_reader, img_parser, pf_handler, args.workers, args.number, args.output)
                executor.run()
