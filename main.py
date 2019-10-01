from train import CNN

if __name__ == '__main__':
	cnn = CNN(model_name='lowdose_cardiac_test_v1',
			  input_patch_shape=(128,128,16),
			  input_channels=2,
			  output_channels=1,
			  batch_size=3,
			  epochs=10,
			  checkpoint_save_rate=2,
			  loss_functions=[['mean_absolute_error',1]],
			  data_pickle='test_dat.pickle',
			  data_folder='/users/claes/projects/Lowdose/Deep/PETrecon/HjerterFDG_mnc'
			  )

	cnn.print_config()

	cnn.train()
