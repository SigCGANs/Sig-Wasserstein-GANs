import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def resample_eICU_patient(pid, resample_factor_in_min, variables, upto_in_minutes):
    """
    Resample a *single* patient.
    """
    pat_df = data[data['patientunitstayid'] == pid ][['observationoffset', 'patientunitstayid']+variables]
    # sometimes it's empty
    if pat_df.empty:
        return None
    if not upto_in_minutes is None:
        pat_df = pat_df.loc[0:upto_in_minutes*60]
    # convert the offset to a TimedeltaIndex (necessary for resampling)
    pat_df.observationoffset = pd.TimedeltaIndex(pat_df.observationoffset, unit='m')
    pat_df.set_index('observationoffset', inplace=True)
    pat_df.sort_index(inplace=True)
    # resample by time
    pat_df_resampled = pat_df.resample(str(resample_factor_in_min) + 'T').median()  # pandas ignores NA in median by default
    # rename pid, cast to int
    pat_df_resampled.rename(columns={'patientunitstayid': 'pid'}, inplace=True)
    pat_df_resampled['pid'] = np.int32(pat_df_resampled['pid'])
    # get offsets in minutes from index
    pat_df_resampled['offset'] = np.int32(pat_df_resampled.index.total_seconds()/60)
    return pat_df_resampled

def generate_eICU_resampled_patients(pids, resample_factor_in_min=15,
        upto_in_minutes=None):
    """
    Generates a dataframe with resampled patients. One sample every "resample_factor_in_min" minutes.
    """

    variables = ['sao2', 'heartrate', 'respiration', 'systemicmean']

    num_pat = 0
    num_miss = 0
    f_miss = open('eICU_proc/' + 'pids_missing_vitals.txt', 'a')
    for pid in pids:    # have to go patient by patient
        pat_df_resampled = resample_eICU_patient(pid, resample_factor_in_min, variables, upto_in_minutes)
        if pat_df_resampled is None:
            f_miss.write(str(pid) + '\n')
            num_miss += 1
            continue
        else:
            if num_pat == 0:
                f = open('eICU_proc/' + 'resampled_pats' + str(resample_factor_in_min) +'min.csv', 'w')
                pat_df_resampled.to_csv(f, header=True, index=False)
            else:
                pat_df_resampled.to_csv(f, header=False, index=False)
            num_pat += 1
        if num_pat % 100 == 0:
            print(num_pat)
            f.flush()
            f_miss.flush()

    print('Acquired data on', num_pat, 'patients.')
    print('Skipped', num_miss, 'patients.')
    return True

def get_cohort_of_complete_downsampled_patients(time_in_hours=4, resample_factor_in_min=15):
    """
    Finds the set of patients that have no missing data during the first "time_in_hours".
    """

    resampled_pats = pd.read_csv('eICU_proc/' + 'resampled_pats' + str(resample_factor_in_min) + 'min.csv')

    time_in_minutes = time_in_hours * 60

    # delete patients with any negative offset
    print('Deleting patients with negative offsets...')
    df_posoffset = resampled_pats.groupby('pid').filter(lambda x: np.all(x.offset >= 0))

    # restrict time consideration
    print('Restricting to offsets below', time_in_minutes)
    df = df_posoffset.loc[df_posoffset.offset <= time_in_minutes]

    #variables = ['sao2', 'heartrate', 'respiration', 'systemicmean']
    variables = ['sao2', 'heartrate', 'respiration']

    # patients with no missing values in those variables (this is slow)
    print('Finding patients with no missing values in', ','.join(variables))
    good_patients = df.groupby('pid').filter(lambda x: np.all(x.loc[:, variables].isnull().sum() == 0))

    # extract the pids, save the cohort
    cohort = good_patients.pid.drop_duplicates()

    if cohort.shape[0] < 2:
        print('ERROR: not enough patients in cohort.', cohort.shape[0])
        return False
    else:
        print('Saving...')
        cohort.to_csv('eICU_proc/' + 'cohort_complete_resampled_pats_' + str(resample_factor_in_min) + 'min.csv', header=False, index=False)
        # save the full data (not just cohort)
        good_patients.to_csv('eICU_proc/'  + 'complete_resampled_pats_' + str(resample_factor_in_min) + 'min.csv', index=False)
        return True

def get_eICU_with_targets(samples, labels, use_age=False, use_gender=False, save=True):
    """
    Load resampled eICU data and get static prediction targets from demographics
    (patients) file
    """
    if use_age: print('Using age!')
    if use_gender: print('Using gender!')
    if save: print('Save!')
    # load resampled eICU data (the labels are the patientunitstayids)

    # load patients static information
    pat_dfs = pd.read_hdf('eICU_proc/patient.h5', mode='r')

    # keep only static information of patients that are in the resampled table
    pat_dfs = pat_dfs[pat_dfs.patientunitstayid.isin(labels)]

    # reordering df to have the same order as samples and labels
    pat_dfs.set_index('patientunitstayid', inplace=True)
    pat_dfs.reindex(labels)

    # target variables to keep. For now we don't use hospitaldischargeoffset since it is the only integer variable.
    #target_vars = ['hospitaldischargeoffset', 'hospitaldischargestatus', 'apacheadmissiondx', 'hospitaldischargelocation', 'unittype', 'unitadmitsource']
    real_vars = ['age']
    binary_vars = ['hospitaldischargestatus', 'gender']
    categorical_vars = ['apacheadmissiondx', 'hospitaldischargelocation', 'unittype', 'unitadmitsource']

    target_vars = categorical_vars + ['hospitaldischargestatus']
    if use_age: target_vars += ['age']
    if use_gender: target_vars += ['gender']

    targets_df = pat_dfs.loc[:, target_vars]

    # remove patients by criteria
            # missing data in any target
    targets_df.dropna(how='any', inplace=True)
    if use_age:
                # age belonw 18 or above 89
        targets_df = targets_df[targets_df.age != '> 89']       # yes, some ages are strings
        targets_df.age = list(map(int, targets_df.age))
        targets_df = targets_df[targets_df.age >= 18]
    if use_gender:
                # remove non-binary genders (sorry!)
        targets_df['gender'] = targets_df['gender'].replace(['Female', 'Male', 'Other', 'Unknown'], [0, 1, -1, -1])
        targets_df = targets_df[targets_df.gender >= 0]
    # record patients to keep
    keep_indices = [i for (i, pid) in enumerate(labels) if pid in targets_df.index]
    assert len(keep_indices) == targets_df.shape[0]
    new_samples = samples[keep_indices]
    new_labels = np.array(labels)[keep_indices]

    # triple check the labels are correct
    assert np.array_equal(targets_df.index, new_labels)

    # getn non-one-hot targets (strings)
    targets = targets_df.values

    # one hot encoding of categorical variables
    dummies = pd.get_dummies(targets_df[categorical_vars], dummy_na=True)
    targets_df_oh = pd.DataFrame()
    targets_df_oh[dummies.columns] = dummies
    # convert binary variables to one-hot, too
    targets_df_oh['hospitaldischargestatus']= targets_df['hospitaldischargestatus'].replace(['Alive', 'Expired'],[1, 0])
    if use_gender:
        targets_df_oh['gender'] = targets_df['gender']  # already binarised
    if use_age:
        targets_df_oh['age'] = 2*targets_df['age']/89 - 1     # 89 is max

    # drop dummy columns marking missing data (they should be empty)
    nancols = [col for col in targets_df_oh.columns if col.endswith('nan')]
    assert np.all(targets_df_oh[nancols].sum() == 0)
    targets_df_oh.drop(nancols, axis=1, inplace=True)
    targets_oh = targets_df_oh.values

    if save:
        # save!
        # merge with training data, for LR saving
        assert new_samples.shape[0] == targets_df_oh.shape[0]
        flat_samples = new_samples.reshape(new_samples.shape[0], -1)
        features_df = pd.DataFrame(flat_samples)
        features_df.index = targets_df_oh.index
        features_df.columns = ['feature_' + str(i) for i in range(features_df.shape[1])]
        all_data = pd.concat([targets_df_oh, features_df], axis=1)
        all_data.to_csv('eICU_proc/eICU_with_targets.csv')

    # do the split
    proportions = [0.6, 0.2, 0.2]
    labels = {'targets': targets, 'targets_oh': targets_oh}
    train_seqs, vali_seqs, test_seqs, labels_split = split(new_samples, proportions, scale=True, labels=labels)
    train_targets, vali_targets, test_targets = labels_split['targets']
    train_targets_oh, vali_targets_oh, test_targets_oh = labels_split['targets_oh']

    return train_seqs, vali_seqs, test_seqs, train_targets, vali_targets, test_targets, train_targets_oh, vali_targets_oh, test_targets_oh

def get_train_data(df, n_hours, seq_length, resample_time,
        future_window_size=0,
        sao2_low=96, heartrate_low=75, respiration_low=15, systemicmean_low=75,
        heartrate_high=100, respiration_high=20, systemicmean_high=100):
    """
    seq_length is how many measurements we use for training
    """
    patients = set(df.pid)
    window_size = int(n_hours*60/resample_time)      # this is how many rows in the window
    X = np.empty(shape=(len(patients), 4*seq_length))  # we have 4*seq_length features
    Y = np.empty(shape=(len(patients), 7))        # we have 7 labels
    i = 0
    kept_patients = [-1337]*len(patients)
    for pat in patients:
        df_pat_withlabels = get_labels(df, pat, seq_length, window_size, 
                future_window_size, sao2_low, heartrate_low, respiration_low, 
                systemicmean_low, heartrate_high, respiration_high, systemicmean_high)
        if df_pat_withlabels is None:
            print('Skipping patient', pat, 'for having too little data')
            continue
        # subset to train period
        df_pat_train = df_pat_withlabels.head(seq_length)
        if df_pat_train.shape[0] < seq_length:
            print('Skipping patient', pat, 'for having too little data')
            continue
        X_pat = df_pat_train[['sao2', 'heartrate', 'respiration', 'systemicmean']].values.reshape(4*seq_length)
        if np.isnan(X_pat).any():
            # this should not happen any more btw
            print('Dropping patient', pat, 'for having NAs')
            # just ignore this row
        else:
            X[i, :] = X_pat
            Y[i, :] = df_pat_train.tail(1)[['low_sao2', 'low_heartrate', 'low_respiration', \
                    'low_systemicmean', 'high_heartrate', 'high_respiration', 'high_systemicmean']].values.reshape(7)*1
            kept_patients[i] = pat
            i += 1
    print('Kept data on', i, 'patients (started with', len(patients), ')')
    # delete the remaining rows
    X = X[:i]
    Y = Y[:i]
    return X, Y, kept_patients
def get_df(resample_time=15, n_hours=1, seq_length=16):
    derived_dir = 'eICU_proc/'
    print('getting patients')
    patients = list(map(int, np.loadtxt(derived_dir + 'cohort_complete_resampled_pats_' + str(resample_time) + 'min.csv')))
    data = derived_dir + 'resampled_pats' + str(resample_time) + 'min.csv'
    print('getting data')
    df = pd.read_csv(data)
    print('subsetting to "complete" patients')
    max_offset = 1.5*(seq_length*resample_time + 60*n_hours)        # for good measure
    print('subsetting by time')
    df = df[df.offset < max_offset]
    # drop patients missing any data in this region (this is slow but it's worth it)
    df = df.groupby('pid').filter(lambda x: np.all(np.isfinite(x.values)) and x.shape[0] > seq_length)
    return df
def get_labels(df, patient, seq_length, window_size, future_window_size,
        sao2_low, heartrate_low, respiration_low, systemicmean_low,
        heartrate_high, respiration_high, systemicmean_high):
    df_pat = df[df.pid == patient]
    if df_pat.shape[0] < seq_length + window_size:
        return None
    df_pat.set_index('offset', inplace=True)
    df_pat.sort_index(inplace=True)
    df_pat_rollmins = df_pat.fillna(1337).rolling(window_size).min()
    df_pat_rollmaxs = df_pat.fillna(-1337).rolling(window_size).max()
    # get thresholds
    low_sao2 = df_pat_rollmins.sao2 < sao2_low
    low_heartrate = df_pat_rollmins.heartrate < heartrate_low
    low_respiration = df_pat_rollmins.respiration < respiration_low
    low_systemicmean = df_pat_rollmins.systemicmean < systemicmean_low
    high_heartrate = df_pat_rollmaxs.heartrate > heartrate_high
    high_respiration = df_pat_rollmaxs.respiration > respiration_high
    high_systemicmean = df_pat_rollmaxs.systemicmean > systemicmean_high
    # extremes
    df_pat_labels = pd.DataFrame({'low_sao2': low_sao2.values, 
                        'low_heartrate': low_heartrate.values,
                        'low_respiration': low_respiration.values,
                        'low_systemicmean': low_systemicmean.values,
                        'high_heartrate': high_heartrate.values,
                        'high_respiration': high_respiration.values,
                        'high_systemicmean': high_systemicmean.values})
    # now we need to align it - first move it back to 0 (subtract window_size),
    # then shift it forward by future_window_size (when we want to make 
    # predictions about)
    df_pat_labels_aligned = df_pat_labels.shift(-window_size + 1 + future_window_size)
    df_pat_labels_aligned.index = df_pat.index
    df_pat_withlabels = pd.concat([df_pat, df_pat_labels_aligned], axis=1)
    return df_pat_withlabels
def gen_data():
    """ just run the whole thing """
    df = get_df(resample_time=15, n_hours=1, seq_length=16)
    X, Y, pids = get_train_data(df, 1, 16, 15, sao2_low=95, respiration_low=13, respiration_high=20, heartrate_low=70, heartrate_high=100, systemicmean_low=70, systemicmean_high=110)
    extreme_heartrate = Y[:, 1] + Y[:, 4]
    extreme_respiration = Y[:, 2] + Y[:, 5]
    extreme_MAP = Y[:, 3] + Y[:, 6]
    Y_OR = np.vstack((extreme_heartrate, extreme_respiration, extreme_MAP)).T
    Y_OR = (Y_OR>0)*1
    pp = [p for p in pids if not p == -1337]
    pre_data = dict()
    pre_data['X'] = X
    pre_data['Y'] = Y
    pre_data['pids'] = pids
    Y_columns = ['low_sao2', 'low_heartrate', 'low_respiration', 'low_systemicmean', 'high_heartrate', 'high_respiration', 'high_systemicmean']
    pre_data['Y_columns'] = Y_columns
    pre_data['Y_ORs'] = Y_OR
    np.save('eICU_task_data2.npy', pre_data)


if __name__=="__main__":
    #data = pd.read_csv('vitalPeriodic.csv')
    generate_eICU_resampled_patients(pids, resample_factor_in_min=15,
        upto_in_minutes=None)
    get_cohort_of_complete_downsampled_patients(time_in_hours=4, resample_factor_in_min=15)
    gen_data()
