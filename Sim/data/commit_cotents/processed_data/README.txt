
The data --> the same data split of ISSTA R4-Table8
Only have training & testing data


Format/Content:

ID :14295be0edba4070a604d93b4e9acb91d5ed39d2
Lable : 0
Twisted Code Changes (real code changes):  ['added_code:  oslo . log > = 1 . 14 . 0 # Apache - 2 . 0 removed_code:  oslo . log > = 1 . 12 . 0 # Apache - 2 . 0']
Commit Logs :  xen : block bootabletestcase from py34 testing the tests use mox which isn't stable with py34 testing , there are race failures with the assertions . this test fails in unexpected ways in unrelated changes so we need to blacklist it from py34 testing . change - id : ieb972f4705254af4b014d79a39bd6d78ad0b6376 closes - bug : #1526369


But in order to do parameter tuning, we use 10% of training data as validation set:
----------------------------------------------
have training & validating & testing data
     75%        5%           20%


Sturcture:

-- qt
        ---- qt_train.pkl : 80% data, whole training data
        |
        ---- qt_test.pkl  : 20% data, hold-out test set
        |
        ---- qt_dict.pkl  : dictionary
        |
        |
        ---- val_train  ----
                           |
                           -----  qt_train.pkl : 75% data, leaving 5% for validation
                           |
                           -----  qt_val.pkl : 5% data, for validation




-- cross_project_train: training data for each project in cross project setting.
                     For instance, jdt_train.pkl --> all training data of (qt, openstack, platform, gerrit, go)
