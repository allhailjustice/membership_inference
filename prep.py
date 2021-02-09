import numpy as np
import pickle
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from itertools import chain


class Visit(object):
    def __init__(self, start_date, end_date, is_inpatient, is_emergency):
        self.start_date = start_date
        self.end_date = end_date
        self.is_inpatient = is_inpatient
        self.is_emergency = is_emergency
        self.codes = set()
        self.age = None
        # self.index = None
        self.interval = None


class Person(object):
    def __init__(self, gender, birth_date):
        self.visits = dict()
        self.gender = gender
        self.birth_date = birth_date


def merge(visits):
    i = 0
    j = len(visits) - 1
    while i < j:
        if visits[i].end_date >= visits[i + 1].start_date:
            visits[i].codes.update(visits[i + 1].codes)
            if visits[i].end_date < visits[i + 1].end_date:
                visits[i].end_date = visits[i + 1].end_date
            # visits[i].index = (visits[i].index or visits[i + 1].index)
            visits[i].is_emergency = (visits[i].is_emergency or visits[i + 1].is_emergency)
            visits[i].is_inpatient = (visits[i].is_inpatient or visits[i + 1].is_inpatient)
            visits.pop(i+1)
            j = len(visits) - 1
        else:
            i += 1
    return visits


def build_matrix(keys, threshold=5):
    code2idx = np.load('../allofus/code2idx.npy', allow_pickle=True).item()
    dx_codes = set([code2idx[code] for code in code2idx.keys() if code.startswith('D')])
    print(len(dx_codes))

    def stay_dic(duration):
        if duration == 0:
            stay = 0
        elif duration == 1:
            stay = 1
        elif duration <= 2:
            stay = 2
        elif duration <= 4:
            stay = 3
        elif duration <= 6:
            stay = 4
        elif duration <= 14:
            stay = 5
        elif duration <= 30:
            stay = 6
        else:
            stay = 7
        return stay

    def age_dic(a):
        if a >= 80:
            return 0
        else:
            return int(a/5)+1

    histories = []
    codes = []
    others = []
    lengths = []
    ages = []

    for patient_id in keys:
        patient = patient_dic[patient_id]
        visits = [visit for visit in patient.visits.values() if len(visit.codes) > 0]
        visits.sort(key=lambda x: x.start_date)
        visits = merge(visits)
        visits = [visit for visit in visits if visit.start_date >= datetime.date(2011,7,1)]
        length = len([visit for visit in visits if visit.end_date < datetime.date(2018,7,1) and
                      visit.start_date >= datetime.date(2016,7,1)])
        history_length = len([visit for visit in visits if visit.end_date < datetime.date(2018,7,1)])
        if history_length > 200:
            visits = visits[history_length-200:]
            history_length = 200
        if length < threshold:
            continue
        history_codes = set([code for code in chain(*[visit.codes for visit in visits[:history_length]]) if code in dx_codes])
        if len(history_codes) == 0:
            continue
        ages.append(int((visits[history_length-1].start_date - patient_dic[patient_id].birth_date).days / 365))
        # for i in range(history_length):
        #     visits[i].duration = (visits[i].end_date - visits[i].start_date).days
        #     visits[i].age = int((visits[i].start_date - patient_dic[patient_id].birth_date).days / 365)
        #     if i == 0:
        #         visits[i].interval = 0
        #     else:
        #         interval = (visits[i].start_date - visits[i - 1].end_date).days
        #         if interval <= 0:
        #             print(interval, patient_id)
        #         elif interval == 1:
        #             visits[i].interval = 1
        #         elif interval == 2:
        #             visits[i].interval = 2
        #         elif interval <= 4:
        #             visits[i].interval = 3
        #         elif interval <= 6:
        #             visits[i].interval = 4
        #         elif interval <= 14:
        #             visits[i].interval = 5
        #         elif interval <= 30:
        #             visits[i].interval = 6
        #         elif interval <= 90:
        #             visits[i].interval = 7
        #         elif interval <= 180:
        #             visits[i].interval = 8
        #         else:
        #             visits[i].interval = 9

        # history_codes = np.array(list(history_codes)).astype('int')-244

        # histories.append(history_codes)
        # tmp_codes = []
        # tmp_others = []
        # for visit in visits[:history_length]:
        #     tmp_codes.append(list(visit.codes))
        #     tmp_age = np.zeros(17)
        #     tmp_age[age_dic(visit.age)] = 1
        #     tmp_stay = np.zeros(8)
        #     tmp_stay[stay_dic(visit.duration)] = 1
        #     tmp_interval = np.zeros(10)
        #     tmp_interval[visit.interval] = 1
        #     tmp_gender = np.zeros(2)
        #     tmp_gender[patient.gender] = 1
        #     tmp_others.append(np.concatenate((tmp_age,tmp_gender,tmp_stay, tmp_interval,
        #                                       np.array([visit.is_emergency],dtype='int'),
        #                                       np.array([visit.is_inpatient],dtype='int'))))
        # if np.max([len(x) for x in tmp_codes]) > 80:
        #     continue
        # codes.append(tmp_codes)
        # others.append(tmp_others)
        # histories.append(np.concatenate([history_codes, -np.ones(115-len(history_codes))],axis=-1))
        # lengths.append(history_length)
    print(np.mean(ages),np.median(ages), max(ages), min(ages))
    print('max_num_visit',np.max([len(x) for x in codes]))
    print('num_range',np.max([len(x) for x in histories]))
    print('max_length_visit',np.max([len(y) for x in codes for y in x]))

    # code_input = []
    # others_input = []
    # for code, other in zip(codes,others):
    #     code = [np.concatenate([np.array(x),-np.ones(80-len(x))],axis=-1) for x in code]
    #     code_input.append(np.concatenate((np.array(code),-np.ones((200-len(code), 80))),axis=0))
    #     others_input.append(np.concatenate((np.array(other),np.zeros((200-len(other), 39))),axis=0))
    #
    # print(len(lengths))
    # np.save('length', np.array(lengths,dtype='int32'))
    # np.save('history', np.array(histories,dtype='int32'))
    # np.save('code',np.array(code_input,dtype='int32'))
    # np.save('others', np.array(others_input,dtype='int32'))


def split_chunk():
    lengths = np.load('length.npy')
    train_idx, test_idx = train_test_split(np.arange(len(lengths)), test_size=0.67)
    test_idx, val_idx = train_test_split(test_idx, test_size=0.5)
    np.save('train_idx',train_idx)
    np.save('test_idx',test_idx)
    np.save('val_idx',val_idx)


if __name__ == '__main__':
    with open('../allofus/patient_dic.pkl','rb') as file:
        patient_dic = pickle.load(file)
        print(len(patient_dic))
        build_matrix(patient_dic.keys())
    # split_chunk()




