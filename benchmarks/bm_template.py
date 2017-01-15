"""
All other benchmarks should be deviated this way so the core functionality
of the benchmarks such as loading or evaluation is not overwritten

Copyright (C) 2017 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import logging

import shutil

import benchmarks.general_utils.io_utils as tl_io
import benchmarks.general_utils.experiments as tl_expt
import benchmarks.bm_registration as bm


def extend_parse(parser):
    """ extent the basic arg parses by some extra required parameters

    :return object:
    """
    # SEE: https://docs.python.org/3/library/argparse.html
    parser.add_argument('--an_executable', type=str, required=True,
                        help='some extra parameters')
    return parser


class BmTemplate(bm.BmRegistration):
    """ a Template showing what method can be overwritten
     * _check_required_params
     * _prepare_single_regist
     * _generate_regist_command
     * _evaluate_single_regist
     * _clean_single_regist

    This benchmarks also presents that method can have registered image
    as output but the transformed landmarks are in different frame
    (reference landmarks in moving image).

    Running in single thread:
    >>> tl_io.create_dir('output')
    >>> params = {'nb_jobs': 1, 'unique': False, 'path_out': 'output',
    ...           'path_cover': 'data/list_pairs_imgs_lnds.csv',
    ...           'an_executable': ''}
    >>> bm = BmTemplate(params)
    >>> bm.run()
    True
    >>> del bm
    >>> shutil.rmtree('output/BmTemplate', ignore_errors=True)
    """

    def _check_required_params(self):
        """ check some extra required parameters for this benchmark """
        super(BmTemplate, self)._check_required_params()
        for param in ['an_executable']:
            assert param in self.params

    def _prepare_single_regist(self, dict_row):
        """ prepare the experiment folder if it is required,
        eq. copy some extra files

        :param dict dict_row: {str: value}, dictionary with regist. params
        :return dict: {str: value}
        """
        logging.debug('.. no preparing before regist. experiment')
        return dict_row

    def _generate_regist_command(self, dict_row):
        """ generate the registration command

        :param dict_row: {str: value}, dictionary with regist. params
        :return: str, the execution string
        """
        target_img = os.path.join(dict_row[bm.COL_REG_DIR],
                              os.path.basename(dict_row[bm.COL_IMAGE_MOVE]))
        target_lnd = os.path.join(dict_row[bm.COL_REG_DIR],
                              os.path.basename(dict_row[bm.COL_POINTS_REF]))
        cmds = ['cp %s %s' % (os.path.abspath(dict_row[bm.COL_IMAGE_MOVE]),
                              target_img),
                'cp %s %s' % (os.path.abspath(dict_row[bm.COL_POINTS_REF]),
                              target_lnd)]
        cmd = ' && '.join(cmds)
        return cmd

    def _evaluate_single_regist(self, dict_row):
        """ evaluate rests of the experiment and identity the registered image
        and landmarks when the process finished

        :param dict_row: {str: value}, dictionary with regist. params
        :return: {str: value}
        """
        logging.debug('.. simulate registration: '
                      'copy the original image and landmarks')
        # detect image
        path_img = os.path.join(dict_row[bm.COL_REG_DIR],
                                os.path.basename(dict_row[bm.COL_IMAGE_MOVE]))
        if os.path.exists(path_img):
            dict_row[bm.COL_IMAGE_REF_TRANS] = path_img
        # detect landmarks
        path_lnd = os.path.join(dict_row[bm.COL_REG_DIR],
                                os.path.basename(dict_row[bm.COL_POINTS_REF]))
        if os.path.exists(path_img):
            dict_row[bm.COL_POINTS_MOVE_TRANS] = path_lnd

        return dict_row

    def _clean_single_regist(self, dict_row):
        """ clean unnecessarily files after the registration

        :param dict_row: {str: value}, dictionary with regist. params
        :return: {str: value}
        """
        logging.debug('.. no cleaning after regist. experiment')
        return dict_row


def main(params):
    """ run the Main of blank experiment

    :param params: {str: value} set of input parameters
    """
    logging.info('running...')
    logging.info(__doc__)
    bm = BmTemplate(params)
    bm.run()
    del bm
    logging.info('Done.')


# RUN by given parameters
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = tl_expt.create_basic_parse()
    parser = extend_parse(parser)
    params = tl_expt.parse_params(parser)
    main(params)