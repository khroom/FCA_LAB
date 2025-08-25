from arl_binarization import BinarizationType
import json


params_for_hist = ['METAL_HEIGHT', 'RAIS_NUM', 'RAIS_TIME', 'LOWR_NUM', 'LOWR_TIME', 'AVG_CTRL_COEFF', 
                    'FEED_AUTO_DUMPS', 'OVERFEED_TIME', 'UNDERFEED_TIME', 'FEED_DOZE_TROLL', 'AVG_RAW_VOLT', 'AVG_VOLT_SETP', 'AVG_VOLT_ADD',
                    'AVG_BACK_EMF', 'POT_AGE', 'METAL_HEIGHT_CALC', 'NUM_UNDERFEEDS', 'NUM_TESTS', 'TEMP_HEAT', 'BATH_HEIGHT',
                    'AVG_VOLT_BUS']


params_for_quartile = ['CR', 'MGF2', 'CAF2', 'TEMPERATURE', 'AE_NUM', 'AE_DUR', 'AVG_AE_VOLT', 'CTRL_MAN_TIME', 'CTRL_LOCK_TIME', 'FEED_MAN_DUMPS',
                        'FEED_MAN_TIME', 'FEED_LOCK_TIME', 'AVG_FEED_SETP', 'TEST_TIME', 'OFF_TIME', 'NOMINAL_TIME', 'AVG_FLUOR_SETP', 'FLUOR_AUTO_DUMPS', 
                        'AVG_NORM_VOLT', 'AVG_NOISE', 'HIGH_NOISE_TIME', 'CARBONDUST', 'WEIGHT_PLAN', 'TAP_RATIO', 'POT_CURR_EFF_LO', 'WEIGHT_POT_LO', 
                        'STARVE_TIME', 'OVERFEED_DUMPS', 'UNDERFEED_DUMPS', 'TEST_DUMPS', 'NOMINAL_DUMPS', 'TAP_TIME', 'TEMP_LIC', 'TEMP_BATH', 
                        'TEMPERATURE_GOAL', 'CR_GOAL', 'AVG_RAW_VOLT_GOAL', 'AVG_ADD_RESET', 'AVG_ADD_TEMP', 'AVG_ADD_NOISE', 'TAP_RESET_INT', 
                        'RES_COVER_INT', 'MEAS_TAP_INT', 'AN_PULL_CNT', 'AVG_ADD_TEMPERATURE', 'AE_NUM_1', 'SI_CRUCE', 'FE_CRUCE', 'FLUOR_DOZE_TROLL', 
                        'VOLT_POD', 'NUM_OVERFEEDS']


def create_model_settings(individual, keep_nan, anomalies_only):
    settings = {}
    for p in params_for_hist:
        settings[p] = {}
        settings[p]['binarization'] = BinarizationType.HISTOGRAMS
        settings[p]['individual'] = individual
        settings[p]['keep_nan'] = keep_nan
        settings[p]['anomalies_only'] = anomalies_only

    for p in params_for_quartile:
        settings[p] = {}
        settings[p]['binarization'] = BinarizationType.QUARTILES
        settings[p]['individual'] = individual
        settings[p]['keep_nan'] = keep_nan
        settings[p]['anomalies_only'] = anomalies_only
    return settings

def dump_model_settings(settings, filename):
    with open(filename,'w') as file:
        json.dump(settings, file, indent=4, default=lambda x : x.value)


def load_model_settings(filename):
    
    def enum_decoder (data):
        if 'binarization' in data:
            data['binarization'] = BinarizationType(data['binarization'])
        return data

    with open(filename,'r') as file:
        settings = json.load (file, object_hook=enum_decoder)

    return settings