import numpy as np
import os
import shutil
import pdb

def remove_zero_row_col(matrix, subject_ID):

    row_zero = np.zeros(shape=(matrix.shape[1], ))
    col_zero = np.zeros(shape=(matrix.shape[0], ))
    row_index = []
    col_index = []

    for row in range(matrix.shape[0]):
        if (matrix[row] == row_zero).all():
            row_index.append(row)

    for col in range(matrix.shape[1]):
        if (matrix[:, col] == col_zero).all():
            col_index.append(col)

    # print 'row:', row_index
    # print 'col:', col_index
    if cmp([41,116],row_index) and cmp([41,116],col_index):
        subject_ID.append(subject)
        # print subject
        # print 'row:', row_index
        # print 'col:', col_index
    matrix = np.delete(matrix, row_index, axis=0)
    matrix = np.delete(matrix, col_index, axis=1)

    if not ((matrix.shape[0]==148 and matrix.shape[1]==148)):
    	print subject
    	print 'new_shape:', matrix.shape

    return matrix


def remove_zero_row_col_signal(matrix, subject_ID):

    row_index = list()

    row_zero = np.zeros(shape=(matrix.shape[1], ))

    for row in range(matrix.shape[0]):
        if (matrix[row] == row_zero).all():
            row_index.append(row)

    matrix = np.delete(matrix, [41,116], axis=0)
    #print 'new_shape:', matrix.shape

    if len(row_index) > 6:
    	print subject
    	print len(row_index)

    return matrix


input_data_folder_path = '../data'
output_data_folder_path = '../data'

SubjID_list = [x for x in os.listdir(input_data_folder_path) if not x.startswith('.')]
SubjID_list.sort()
print SubjID_list
print len(SubjID_list)
# SubjID_list =['002_S_0413']
# exit()

subject_ID = []

for subject in SubjID_list:
	# print(subject)

	if os.path.exists(input_data_folder_path + '/' + subject + '/' + 'nonzero_fmri_average_signal'):
		shutil.rmtree(input_data_folder_path + '/' + subject + '/' + 'nonzero_fmri_average_signal')
	os.makedirs(input_data_folder_path + '/' + subject + '/' + 'nonzero_fmri_average_signal')

	if os.path.exists(input_data_folder_path + '/' + subject + '/' + 'nonzero_fmri_average_signal_normalized1'):
		shutil.rmtree(input_data_folder_path + '/' + subject + '/' + 'nonzero_fmri_average_signal_normalized1')
	os.makedirs(input_data_folder_path + '/' + subject + '/' + 'nonzero_fmri_average_signal_normalized1')

	if os.path.exists(input_data_folder_path + '/' + subject + '/' + 'nonzero_fmri_average_signal_normalized2'):
		shutil.rmtree(input_data_folder_path + '/' + subject + '/' + 'nonzero_fmri_average_signal_normalized2')
	os.makedirs(input_data_folder_path + '/' + subject + '/' + 'nonzero_fmri_average_signal_normalized2')

    # adj matrix part
	fiber_matrix_path = input_data_folder_path + '/' + subject +'/' + 'common_fiber_matrix.txt'
	fiber_matrix = np.loadtxt(fiber_matrix_path)
	fiber_matrix = remove_zero_row_col(fiber_matrix, subject_ID)
	np.save(output_data_folder_path + '/' +subject + "/nonzero_common_fiber_matrix.npy", fiber_matrix)
	np.savetxt(output_data_folder_path + '/' +subject + "/nonzero_common_fiber_matrix.txt", fiber_matrix, fmt='%-8.1f')

    # feature matrix part
	fmri_matrix_folder_path0 = input_data_folder_path + '/' + subject + '/' + 'fmri_average_signal'
	fmri_matrix_folder_path1 = input_data_folder_path + '/' + subject + '/' + 'fmri_average_signal_normalized1'
	fmri_matrix_folder_path2 = input_data_folder_path + '/' + subject + '/' + 'fmri_average_signal_normalized2'
	for time_num in range(191):
		fmri_matrix_path = fmri_matrix_folder_path0 + '/' + 'average_fmri_feature_matrix_' + str(time_num) + '.npy'
		if not os.path.isfile(fmri_matrix_folder_path0 + '/' + 'average_fmri_feature_matrix_' + str(time_num) + '.npy'):
			fmri_matrix_path = fmri_matrix_folder_path0 + '/' + 'raw_fmri_feature_matrix_' + str(time_num) + '.npy'
		fmri_matrix = np.load(fmri_matrix_path)
		fmri_matrix = fmri_matrix.reshape(-1, 1)
		#db.set_trace()
		fmri_matrix = remove_zero_row_col_signal(fmri_matrix, subject_ID)
		np.save(input_data_folder_path + '/' + subject + '/' + 'nonzero_fmri_average_signal' + '/raw_fmri_feature_matrix_' + str(time_num) + ".npy", fmri_matrix)
		np.savetxt(input_data_folder_path + '/' + subject + '/' + 'nonzero_fmri_average_signal' + "/raw_fmri_feature_matrix_" + str(time_num) + ".txt",fmri_matrix, fmt='%8.8f')

		fmri_matrix_path = fmri_matrix_folder_path1 + '/' + 'average_fmri_feature_matrix_' + str(time_num) + '.npy'
		if not os.path.isfile(fmri_matrix_folder_path1 + '/' + 'average_fmri_feature_matrix_' + str(time_num) + '.npy'):
			fmri_matrix_path = fmri_matrix_folder_path1 + '/' + 'raw_fmri_feature_matrix_' + str(time_num) + '.npy'
		fmri_matrix = np.load(fmri_matrix_path)
		fmri_matrix = fmri_matrix.reshape(-1, 1)
		fmri_matrix = remove_zero_row_col_signal(fmri_matrix, subject_ID)
		np.save(input_data_folder_path + '/' + subject + '/' + 'nonzero_fmri_average_signal_normalized1' + "/raw_fmri_feature_matrix_" + str(time_num) + ".npy", fmri_matrix)
		np.savetxt(input_data_folder_path + '/' + subject + '/' + 'nonzero_fmri_average_signal_normalized1' + "/raw_fmri_feature_matrix_" + str(time_num) + ".txt",fmri_matrix, fmt='%8.8f')
		
		fmri_matrix_path = fmri_matrix_folder_path2 + '/' + 'average_fmri_feature_matrix_' + str(time_num) + '.npy'
		if not os.path.isfile(fmri_matrix_folder_path2 + '/' + 'average_fmri_feature_matrix_' + str(time_num) + '.npy'):
			fmri_matrix_path = fmri_matrix_folder_path2 + '/' + 'raw_fmri_feature_matrix_' + str(time_num) + '.npy'
		fmri_matrix = np.load(fmri_matrix_path)
		fmri_matrix = fmri_matrix.reshape(-1, 1)
		fmri_matrix = remove_zero_row_col_signal(fmri_matrix, subject_ID)
		np.save(input_data_folder_path + '/' + subject + '/' + 'nonzero_fmri_average_signal_normalized2' + "/raw_fmri_feature_matrix_" + str(time_num) + ".npy", fmri_matrix)
		np.savetxt(input_data_folder_path + '/' + subject + '/' + 'nonzero_fmri_average_signal_normalized2' + "/raw_fmri_feature_matrix_" + str(time_num) + ".txt",fmri_matrix, fmt='%8.8f')

print subject_ID

