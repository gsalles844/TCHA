# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:33:21 2024

@author: gregoires
"""
import argparse
from copy import deepcopy
from functools import reduce, wraps
import json
import logging as log
import os
import shutil
import sys
import time
import traceback as tb


from Evaluate import interpolateTracks

from ProcessMultipliers.processMultipliers import LocalWinds

from Utilities.config import ConfigParser
from Utilities.files import flStartLog
from Utilities.loadData import format_RSMC_track_files


def timer(f):
    """
    Basic timing functions for entire process
    """
    @wraps(f)
    def wrap(*args, **kwargs):
        t1 = time.time()
        res = f(*args, **kwargs)

        tottime = time.time() - t1
        msg = "%02d:%02d:%02d " % \
          reduce(lambda ll, b : divmod(ll[0], b) + ll[1:],
                        [(tottime,), 60, 60])

        log.info("Time for {0}: {1}".format(f.__name__, msg) )
        return res

    return wrap

def doOutputDirectoryCreation(configFile):
    """
    Create all the necessary output folders.

    :param str configFile: Name of configuration file.
    :raises OSError: If the directory tree cannot be created.

    """
    config = ConfigParser()
    config.read(configFile)

    outputPath = config.get('Output', 'Path')

    log.info('Output will be stored under %s', outputPath)

    subdirs = ['tracks', 'windfield']

    if not os.path.isdir(outputPath):
        try:
            os.makedirs(outputPath)
        except OSError:
            raise
    for subdir in subdirs:
        if not os.path.isdir(os.path.realpath(os.path.join(outputPath, subdir))):
            try:
                os.makedirs(os.path.realpath(os.path.join(outputPath, subdir)))
            except OSError:
                raise

def doCleanupAction(outpath, is_exception=False):
    '''
    Small collection of functions to clean up after the run. The main purpose is
    to gather files under the same directory `data` in the `output` folder. Other
    files (both temporary and unused datasets) are removed.

    Parameters
    ----------
    outpath : str
        Path-like str pointing to the output directory.
        
    is_exception : bool, optional
        Boolean used to define how much clean-up is necessary. Passing a True
        value will result in both the deletion of temporary files and 
        displacement of outputs datasets to the `output/data` directory. 
        The default is False.

    '''
    def remove_temporary_dir(tmp_path):
        if os.path.isdir(tmp_path):
            shutil.rmtree(tmp_path)
    
    def sort_output(fpath):
        dname = 'data'
        new_out = os.path.join(fpath, dname)
        if not os.path.isdir(new_out):
            os.mkdir(new_out)
        
        flist = os.listdir(fpath)
        for file in flist:
            if not os.path.isdir(os.path.join(fpath, file)):
                continue
            elif file == dname:
                continue
            else:
                pass
        
            filepath = os.path.join(fpath, file)
            files_to_keep = ['bear_prj.tif', 'gust_prj.tif', \
                             'local_wind.tif', 'region_wind.tif']
            tiffList = [f for f in os.listdir(filepath) if f in files_to_keep]
            
            for f in tiffList:
                src = os.path.join(filepath, f)
                
                if 'region_wind' in f:
                    dst = os.path.join(new_out, f)
                else:
                    dst = os.path.join(new_out, file + '_' + f)
                shutil.move(src, dst)
                
            shutil.rmtree(filepath)
    #TODO: faire un truc plus propre la dessus
    def move_input_to_output(fpath):
        root = os.path.split(os.path.dirname(__file__))[0]
        for file in os.listdir(root):
            if file.endswith('.csv') or file.endswith('.json'):
                f_input = os.path.join(root, file)
                f_output = os.path.join(fpath, file)
                shutil.move(f_input, f_output)
        return
    
    # Better make sure every processes are done with their files opening / closing
    from time import sleep
    sleep(1)
    # In this case, we shouldnt have any results so we only remove the tmp file
    tmp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'tmp'))
    if is_exception:
        remove_temporary_dir(tmp_path)
    else:
        remove_temporary_dir(tmp_path)
        sort_output(outpath)
        move_input_to_output(outpath)
    
    
    
    


class ConfigPlugIn:
    tcevent_sections = ['DataProcess', 'WindfieldInterface', 'Input', 'Timeseries', 'Output', 'Logging', 'IBTrACS', 'RSMC', 'Region']
    local_winds_sections = ['Input', 'Output', 'Logging']
    
    def __init__(self, json_config):
        self.config = self.read_json_file(json_config)
        self.default_tcevent_config = self.relpath(['..', 'ancillary_data', '00_tcevent_configuration', 'default_tcevent_configuration_file.ini'])    
        self.default_local_wind_config = self.relpath(['..', 'ancillary_data', '00_tcevent_configuration', 'default_local_winds_configuration_file.ini'])


    def generate_tcevent_config(self):
        '''
        Create the configuration file required for the tcevent module. Generated
        files are based on the initial parameters provided by the user and a default
        configuration file.
        
        Only the paths are added to the default configuration file. All the other
        values are neither modified nor defined.
        '''
        config = ConfigParser()
        config.read(self.default_tcevent_config)
        
        
        tmp = self.make_tmp_dir()
        
        if self.config['tracks_source'] in ['RSMC']:
            tracks_file = self.relpath(['tmp', 'cyclone_tracks.csv'])
            dfmt = config.get('RSMC', 'DateFormat')
            
            tmp2 = self.relpath(['..', 'ancillary_data', self.config['country_code'] + '_wind_multipliers', 'EEZ'])
            fname = [ f for f in os.listdir(tmp2) if f.endswith('.shp') ][0]
            eez_path = os.path.join(tmp2, fname)
            
            gridLim = format_RSMC_track_files(self.config['tracks_file'], tracks_file, eez_path, dfmt)
            
        elif self.config['tracks_source'] == 'IBTrACS':
            tracks_file = self.config['tracks_file']
        else:
            raise ValueError('{} is not an accepted parameter. Sources can either be "RSMC" or "IBTraCS"'.format(self.config['tracks_source']))

        fpath = os.path.join(tmp, 'tmp_tcevent_config.ini')
        log_out = self.get_output_dir([self.config['cyclone_name'] + '.log'])

        # Seting up the paths
        config.set('DataProcess', 'InputFile', tracks_file)
        config.set('DataProcess', 'Source', self.config['tracks_source'])
        config.set('Output', 'Path', tmp) 
        config.set('Logging', 'LogFile', log_out)
        config.set('Region', 'gridLimit', gridLim)
        
        # Write it into a text file
        self.write_configuration_file(config, fpath, module='tcevent')
        return config, fpath

    def generate_local_winds_config(self, gust_file, tmp_dir):
        '''
        Create the configurations files required for the processMultipliers.py 
        script based the initial parameters provided by the user and a default
        configuration file.
        
        Only the paths are added to the default configuration file. All the other
        values are neither modified nor defined.
        '''
        cnt_dir = '{}_wind_multipliers'.format(self.config['country_code'])
        wm_multipliers_dir = self.relpath(['..', 'ancillary_data', cnt_dir])
        flist = os.listdir(wm_multipliers_dir)
        
        log_out = self.get_output_dir([self.config['cyclone_name'] + '.log'])
        
        configs = []
        for file in flist:
            if file.endswith('.tif'):
                fname = file.split('.tif')[0]
                
                config = ConfigParser()
                config.read(self.default_local_wind_config)
                
                # Setting up the configuration file
                fpath = self.relpath([wm_multipliers_dir, file])
                
                config.set('Input', 'Multipliers', fpath)
                config.set('Input', 'Gust_file', gust_file)
                config.set('Output', 'Working_dir', self.get_output_dir([fname]))
                config.set('Output', 'path', self.get_output_dir([fname]))
                config.set('Logging', 'LogFile', log_out)
                
                # Writting the configuration file
                cpath = os.path.join(tmp_dir, fname + '.ini')
                self.write_configuration_file(config, cpath, module='local_winds')
                
                out = (deepcopy(config), cpath)
                configs.append(out)
                del config
        return configs
    
    def write_configuration_file(self, config, fpath, module):
        '''
        Method used to produce the configuration files fed to the different TCRM
        modules. 

        Parameters
        ----------
        config : _ConfigParser like Object (see Utilities/config.py)
            Object containing the configuration for a given module.
            
        fpath : Str
            Path-like object pointing to the configuration file that will be 
            generated.
            
        module : Str
            Name of the module for which the configuration is to be created. 
            Currently only two options are accepted: `tcevent` or `local_winds`.

        '''
        if module == 'tcevent':
            sections = self.tcevent_sections
        elif module == 'local_winds':
            sections = self.local_winds_sections
        else:
            raise ValueError('{} is not an accepted parameter.'.format(module) + \
                             'Module can either be "tcevent" or "local_winds"')
            
        # Creating the content of the configuration file, a list of strings
        # with a newline character (\n) at the end of each string.
        content = []
        for section in sections:
            items = config.items(section)
            content.append('[{}]\n'.format(section))

            for item in items:
                content.append('{}={}\n'.format(item[0], item[1]))
        
        # Write the configuration into the temporary folder
        with open(fpath, 'w') as f:
            f.writelines(content)
    
    def get_output_dir(self, items=[]):
        '''
        Create a relative path (starting from the location of the main.py file)
        toward the output directory for a given TC event. Additional subfolders
        can be passed in a list using the `items` argument.
        
        Example:
            ConfigRun = ConfigPlugIn(json_config=args.config_file)
            outpath = ConfigRun.get_output_dir(['Data', 'subfolder_1'])
            
        Will return '..\output\TC_NAME\Data\subfolder_1'
        Where TC_NAME is the TC name defined in the Jason configuration file.
        
        Parameters
        ----------
        items : List, optional
            List of strings. Each element must correspond to the name of a 
            subfolder present in the 'output\TC_NAME' directory.
            The default is [].

        Returns
        -------
        out : Str
            Path-like string pointing to the output directory.

        '''
        return self.relpath(['..', 'output', self.config['cyclone_name']] + items)
    
    def make_tmp_dir(self):
        '''
        Create a temporary folder in the `tcrm` directory to store the configuration
        files. Said directory is removed at the end of the run and all every
        files contained in it are deleted.

        Returns
        -------
        out : Str
            Path-like string pointing to the temporary directory.

        '''
        out = os.path.abspath(os.path.join(os.path.dirname(__file__), 'tmp'))
        if not os.path.exists(out):
            os.mkdir(out)
        return out
    
    @staticmethod
    def relpath(items):
        '''
        Create a relative path starting from the position of the tcrm directory
        and pointing to the last element present in items.
        
        Parameters
        ----------
        items : List, optional
            List of strings.
        '''
        return os.path.relpath(os.path.join(*items))
    
    @staticmethod
    def read_json_file(fpath):
        '''
        Read a Jason file and return its content in a dictionary.
        '''
        with open(fpath, 'r') as f:
            data = f.read()
        return json.loads(data)
    
    



def setup():
    """
    Parse the command line arguments and call the modules one after another.
    
    The first module run, `tcevent`, is used to generate the regional winds from
    the TC tracks file.
    
    The second module, `local_winds`, is used to generate the local winds over
    land (and water) by making use of the Wind Multipliers datasets (present in
    the ancillary_data directory) and the previously estimated regional winds.

    """
    # Switch off minor warning messages
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="pytz")
    warnings.filterwarnings("ignore", category=UserWarning, module="numpy")
    warnings.filterwarnings("ignore", category=UserWarning,
                            module="matplotlib")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # Parsing the arguments passed with the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file',
                        help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Parsing the Jason conf. file and creating the one required to run the 
    # `tcevent` module
    ConfigRun = ConfigPlugIn(json_config=args.config_file)
    config, configFile = ConfigRun.generate_tcevent_config()
    
    # Set-up the log file    
    logfile = config.get('Logging','LogFile')
    logLevel = config.get('Logging', 'LogLevel')
    verbose = config.getboolean('Logging', 'Verbose')
    datestamp = config.getboolean('Logging', 'Datestamp')
    flStartLog(logfile, logLevel, verbose, datestamp)
    
    # Launch the first module (tcevent)
    try:
        windfieldPath = tcevent(configFile)
    except Exception:
        # Catch any exceptions that occur and log them (nicely):
        tblines = tb.format_exc().splitlines()
        for line in tblines:
            log.critical(line.lstrip())
        doCleanupAction(None, is_exception=True)
        raise
    
    log.info('End of the tcevent module')
    log.info('Starting to work on the estimation of local winds')
    # sys.exit(0)
    # Creating a path pointing to the regional wind dataset to use it as inputs
    gust_file = [ f for f in os.listdir(windfieldPath) if f.startswith('gust') ][0]
    windfieldPath = os.path.join(windfieldPath, gust_file)
    configs = ConfigRun.generate_local_winds_config(windfieldPath, os.path.split(configFile)[0])
    
    log.info('A total of {} domains were found'.format(len(configs)))
    
    # Launch the second module (local_winds)
    try:
        local_winds(configs)
    except Exception:
        # Catch any exceptions that occur and log them (nicely):
        tblines = tb.format_exc().splitlines()
        for line in tblines:
            log.critical(line.lstrip())
        doCleanupAction(None, is_exception=True)
        raise
    
    log.info('End of the local_winds module')
    log.info('Starting the clean-up actions')
    
    # Cleaning up the temporary folder / outputs
    outpath = ConfigRun.get_output_dir()
    doCleanupAction(outpath, is_exception=False)
    return




def tcevent(configFile):
    '''
    `tcevent` module. 
    This module parse the tracks file and interpolate it along the track to 
    generate a regional wind dataset.

    Parameters
    ----------
    configFile : Str
        Path pointing to the configuration file.

    Returns
    -------
    windfieldPath : Str
        Path pointing to the generated regional wind dataset.
    trackPath : Str
        Path poiting to the TC tracks dataset.

    '''
    config = ConfigParser()
    config.read(configFile)
    
    doOutputDirectoryCreation(configFile)
    
    trackFile = config.get('DataProcess', 'InputFile')
    source = config.get('DataProcess', 'Source')
    delta = 1/12.
    outputPath = os.path.join(config.get('Output','Path'), 'tracks')
    outputTrackFile = os.path.join(outputPath, "tracks.interp.nc")
    
    # This will save interpolated track data in TCRM format:
    interpTrack = interpolateTracks.parseTracks(configFile, trackFile,
                                                source, delta,
                                                outputTrackFile,
                                                interpolation_type='akima')
    
    import tcrm.wind as wind
    windfieldPath, _ = wind.run(configFile, None)
    return windfieldPath

def local_winds(configs):
    '''
    `local_winds` module.
    This module makes use of the previously estimated regional wind and Wind
    Multipliers (ancillary data) to estimate the local winds over land.

    Parameters
    ----------
    configs : List
        List of tupples. Each tupple contains the configuration object and the
        path pointing to the associated configuration file.
    '''
    for item in configs:
        name = os.path.split(item[1])[1]
        outpath = item[0].get('Output', 'Working_dir')
        log.info('Generating the Local winds for {}'.format(name))
        
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        
        if 'config' in globals():
            del config
            
        # Initialize the object by passing the configuration file used to
        # define the subdomain we are working on
        Winds = LocalWinds(item[1])
        Winds.main()
    



if __name__ == "__main__":
    setup()