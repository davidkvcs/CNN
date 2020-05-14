#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os

plt.close('all')

def jaj_plot(im, save_plot = 1, mapc = 0, lim = 0, alph = 1, axis = 1, grid = 1, colorbar = 1, colorbar_lab = ' '):
	if lim == 0:
		lim 	= im.min(), im.max()
		print(lim)
	elif lim == "soft":
		lim 	= -135, 215
		print(lim)
	if mapc == 0:
		mapc 	= "gray"
		print(mapc)
	fov = 500.
	v = (0, round(((fov/512)*im.shape[1])), 0, round(((fov/512)*im.shape[0]))) #rescaling the axis
	ax = plt.imshow(im, cmap = mapc, clim = lim, alpha = alph, extent = v)
	if colorbar == 1:
		cbar = plt.colorbar(ax)
		cbar.set_label(colorbar_lab)#, rotation = 90)
	#set axes
	plt.grid(color='gray', alpha = 0.5 , linestyle='-', linewidth=1.5)
	if axis == 0:
		plt.axis('off')
	if grid == 0:
		plt.grid('off')
	if save_plot == 1:
		print('saving plot at ' + os.getcwd())
		plt.savefig(fname = 'test.png')
	return ax
