# -*- coding: utf-8 -*-
'''
/**************************************************************************************************************************
 SemiAutomaticClassificationPlugin

 The Semi-Automatic Classification Plugin for QGIS allows for the supervised classification of remote sensing images, 
 providing tools for the download, the preprocessing and postprocessing of images.

							 -------------------
		begin				: 2012-12-29
		copyright		: (C) 2012-2021 by Luca Congedo
		email				: ing.congedoluca@gmail.com
**************************************************************************************************************************/
 
/**************************************************************************************************************************
 *
 * This file is part of Semi-Automatic Classification Plugin
 * 
 * Semi-Automatic Classification Plugin is free software: you can redistribute it and/or modify it under 
 * the terms of the GNU General Public License as published by the Free Software Foundation, 
 * version 3 of the License.
 * 
 * Semi-Automatic Classification Plugin is distributed in the hope that it will be useful, 
 * but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
 * FITNESS FOR A PARTICULAR PURPOSE. 
 * See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with 
 * Semi-Automatic Classification Plugin. If not, see <http://www.gnu.org/licenses/>. 
 * 
**************************************************************************************************************************/

'''



cfg = __import__(str(__name__).split('.')[0] + '.core.config', fromlist=[''])

class Sentinel2Tab:

	def __init__(self):
		pass
		
	# Sentinel-2 input
	def inputSentinel(self):
		i = cfg.utls.getExistingDirectory(None , cfg.QtWidgetsSCP.QApplication.translate("semiautomaticclassificationplugin", "Select a directory"))
		cfg.ui.S2_label_86.setText(str(i))
		self.populateTable(i)
		# logger
		cfg.utls.logCondition(str(__name__) + '-' + str(cfg.inspectSCP.stack()[0][3])+ ' ' + cfg.utls.lineOfCode(), str(i))
		
	# XML input MTD_SAFL1C
	def inputXML2(self):
		m = cfg.utls.getOpenFileName(None , cfg.QtWidgetsSCP.QApplication.translate("semiautomaticclassificationplugin", "Select a XML file"), "", "XML file .xml (*.xml)")
		cfg.ui.S2_label_94.setText(str(m))
		if len(cfg.ui.S2_label_86.text()) > 0:
			self.populateTable(cfg.ui.S2_label_86.text())
		# logger
		cfg.utls.logCondition(str(__name__) + '-' + str(cfg.inspectSCP.stack()[0][3])+ ' ' + cfg.utls.lineOfCode(), str(m))
			
	# populate table
	def populateTable(self, input, batch = 'No'):
		check = 'Yes'
		sat = 'Sentinel-2A'
		# dt = ''
		# sE = ''
		# esd = ''
		dEsunB = {'ESUN_BAND01': 1913.57, 'ESUN_BAND02': 1941.63, 'ESUN_BAND03': 1822.61, 'ESUN_BAND04': 1512.79, 'ESUN_BAND05': 1425.56, 'ESUN_BAND06': 1288.32, 'ESUN_BAND07': 1163.19, 'ESUN_BAND08': 1036.39, 'ESUN_BAND8A': 955.19, 'ESUN_BAND09': 813.04, 'ESUN_BAND10': 367.15, 'ESUN_BAND11': 245.59, 'ESUN_BAND12': 85.25}
		quantVal = 10000
		dU = 1
		cfg.ui.S2_satellite_lineEdit.setText(sat)
		cfg.ui.S2_product_lineEdit.setText('')
		cfg.ui.date_lineEdit_3.setText('')
		l = cfg.ui.sentinel_2_tableWidget
		cfg.utls.setColumnWidthList(l, [[0, 450]])
		cfg.utls.clearTable(l)
		inp = input
		if len(inp) == 0:
			cfg.mx.msg14()
		else:
			if batch == 'No':
				cfg.uiUtls.addProgressBar()
			if len(cfg.ui.S2_label_94.text()) == 0:
				MTD_SAFL1C = None
				for f in cfg.osSCP.listdir(inp):
					if f.lower().endswith('.xml') and 'mtd_msil1c' in f.lower() or 'mtd_safl1c' in f.lower()  or 'mtd_msil2a' in f.lower():
							MTD_SAFL1C = inp + '/' + str(f)
			else:
				MTD_SAFL1C = cfg.ui.S2_label_94.text()
			# open metadata file
			try:
				if MTD_SAFL1C is not None:
					doc2 = cfg.minidomSCP.parse(MTD_SAFL1C)
					SPACECRAFT_NAME = doc2.getElementsByTagName('SPACECRAFT_NAME')[0]
					sat = SPACECRAFT_NAME.firstChild.data
					cfg.ui.S2_satellite_lineEdit.setText(sat)
					PRODUCT_START_TIME = doc2.getElementsByTagName('PRODUCT_START_TIME')[0]
					date = PRODUCT_START_TIME.firstChild.data
					cfg.ui.date_lineEdit_3.setText(date.split('T')[0])
					PRODUCT_TYPE = doc2.getElementsByTagName('PRODUCT_TYPE')[0]
					productType = PRODUCT_TYPE.firstChild.data
					cfg.ui.S2_product_lineEdit.setText(productType)
					paramU = doc2.getElementsByTagName('U')[0]
					dU = paramU.firstChild.data
					QUANTIFICATION_VALUE = doc2.getElementsByTagName('QUANTIFICATION_VALUE')[0]
					quantVal = QUANTIFICATION_VALUE.firstChild.data
					SOLAR_IRRADIANCE = doc2.getElementsByTagName('SOLAR_IRRADIANCE')
					dEsunB = {'ESUN_BAND01': SOLAR_IRRADIANCE[0].firstChild.data, 'ESUN_BAND02': SOLAR_IRRADIANCE[1].firstChild.data, 'ESUN_BAND03': SOLAR_IRRADIANCE[2].firstChild.data, 'ESUN_BAND04': SOLAR_IRRADIANCE[3].firstChild.data, 'ESUN_BAND05': SOLAR_IRRADIANCE[4].firstChild.data, 'ESUN_BAND06': SOLAR_IRRADIANCE[5].firstChild.data, 'ESUN_BAND07': SOLAR_IRRADIANCE[6].firstChild.data, 'ESUN_BAND08': SOLAR_IRRADIANCE[7].firstChild.data, 'ESUN_BAND8A': SOLAR_IRRADIANCE[8].firstChild.data, 'ESUN_BAND09': SOLAR_IRRADIANCE[9].firstChild.data, 'ESUN_BAND10': SOLAR_IRRADIANCE[10].firstChild.data, 'ESUN_BAND11': SOLAR_IRRADIANCE[11].firstChild.data, 'ESUN_BAND12': SOLAR_IRRADIANCE[12].firstChild.data}
			except Exception as err:
				# logger
				cfg.utls.logCondition(str(__name__) + '-' + str(cfg.inspectSCP.stack()[0][3])+ ' ' + cfg.utls.lineOfCode(), ' ERROR exception: ' + str(err))
				if batch == 'No':
					cfg.uiUtls.removeProgressBar()
				#if '2A' in cfg.ui.S2_product_lineEdit.text():
				#	pass
				#else:
				#	cfg.mx.msgWar21()
				#	check = 'No'
			# bands
			dBs = {}
			bandNames = []
			for f in cfg.osSCP.listdir(inp):
				if f.lower().endswith('.jp2') or f.lower().endswith('.tif'):
					# check band number
					bNmb = str(f[-6:-4])
					if str(bNmb).lower() in ['01','02','03','04','05','06','07','08','8a','09','10','11','12']:
						bandNames.append(f)
			# add band items to table
			b = 0
			for band in sorted(bandNames):
				l.insertRow(b)
				l.setRowHeight(b, 20)
				cfg.utls.addTableItem(l, band, b, 0, 'No')
				cfg.utls.addTableItem(l, quantVal, b, 1)
				eS = str(dEsunB['ESUN_BAND' + str(band[-6:-4])])				
				cfg.utls.addTableItem(l, eS, b, 2)
				b = b + 1
			if batch == 'No':
				cfg.uiUtls.removeProgressBar()
			
	# edited satellite
	def editedSatellite(self):
		sat = cfg.ui.S2_satellite_lineEdit.text()
		if str(sat).lower() in ['sentinel_2a', 'sentinel-2a', 'sentinel_2b', 'sentinel-2b']:
			cfg.ui.S2_satellite_lineEdit.setStyleSheet('color : black')
		else:
			cfg.ui.S2_satellite_lineEdit.setStyleSheet('color : red')
	
	# remove band
	def removeHighlightedBand(self):
		l = cfg.ui.sentinel_2_tableWidget
		cfg.utls.removeRowsFromTable(l)

	# edited cell
	def editedCell(self, row, column):
		if column != 0:
			l = cfg.ui.sentinel_2_tableWidget
			val = l.item(row, column).text()
			try:
				float(val)
			except:
				l.item(row, column).setText('')

	# sentinel-2 output
	def performSentinelConversion(self):
		if len(cfg.ui.S2_label_86.text()) == 0:
			cfg.mx.msg14()
		else:
			o = cfg.utls.getExistingDirectory(None , cfg.QtWidgetsSCP.QApplication.translate('semiautomaticclassificationplugin', 'Select a directory'))
			if len(o) == 0:
				cfg.mx.msg14()
			else:
				self.sentinel2(cfg.ui.S2_label_86.text(), o)
				# logger
				cfg.utls.logCondition(str(__name__) + '-' + str(cfg.inspectSCP.stack()[0][3])+ ' ' + cfg.utls.lineOfCode(), "Perform sentinel-2 conversion: " + str(cfg.ui.S2_label_86.text()))
		
	# sentinel-2 conversion
	def sentinel2(self, inputDirectory, outputDirectory, batch = 'No', bandSetNumber = None):
		if bandSetNumber is None:
			bandSetNumber = cfg.bndSetNumber
		if bandSetNumber >= len(cfg.bandSetsList):
			cfg.mx.msgWar25(bandSetNumber + 1)
			return 'No'
		cfg.uiUtls.addProgressBar()
		# disable map canvas render for speed
		if batch == 'No':
			cfg.cnvs.setRenderFlag(False)
		self.sA = ''
		self.eSD = ''
		sat = cfg.ui.S2_satellite_lineEdit.text()
		productType = cfg.ui.S2_product_lineEdit.text()
		dt = cfg.ui.date_lineEdit_3.text()
		if str(sat) == "":
			# logger
			cfg.utls.logCondition(str(__name__) + '-' + str(cfg.inspectSCP.stack()[0][3])+ ' ' + cfg.utls.lineOfCode(), ' No satellite error')
			if batch == 'No':
				cfg.uiUtls.removeProgressBar()
				cfg.cnvs.setRenderFlag(True)
			cfg.mx.msgErr37()
			return 'No'
		cfg.uiUtls.updateBar(5)	
		l = cfg.ui.sentinel_2_tableWidget
		inp = inputDirectory
		out = outputDirectory
		cfg.utls.makeDirectory(out)
		# name prefix
		pre = "RT_"
		# input bands
		c = l.rowCount()
		# temp raster
		tempRasterList = []
		# output raster list
		outputRasterList = []
		# band list
		bandSetList = []
		bandSetNameList = []
		inputList = []
		resampleList = []
		functionList = []
		variableList = []
		# No data value
		if cfg.ui.S2_nodata_checkBox.isChecked() is True:
			nD = cfg.ui.S2_nodata_spinBox.value()
		else:
			nD = cfg.NoDataVal
		for i in range(0, c):
			if cfg.actionCheck == 'Yes':
				bNmb = str(l.item(i,0).text()[-6:-4])
				processBand = 'Yes'
				if bNmb in ['01', '09', '10']:
					if cfg.ui.preprocess_b_1_9_10_checkBox.isChecked() is True:
						processBand = 'Yes'
					else:
						processBand = 'No'
				if processBand == 'Yes':
					bNm = str(l.item(i,0).text()[0:-4])
					inputRaster = inp + '/' + l.item(i,0).text()
					oNm = pre + bNm
					outputRaster = out + '/' + oNm + '.tif'
					quantificationValue = float(l.item(i,1).text())
					radC = 1 / float(quantificationValue)
					ESUN = float(l.item(i,2).text())
					if cfg.ui.DOS1_checkBox_S2.isChecked() is False or (cfg.ui.DOS1_checkBox_S2.isChecked() is True and '2A' in productType):
						# calculate reflectance = DN / quantVal
						e = 'cfg.np.clip(cfg.np.where(raster == ' + str(nD) + ', ' + str(cfg.NoDataVal) + ', ( raster /' + str(quantificationValue) + ') ), 0, 1)'
						functionList.append(e)
						variableList.append(['raster'])
						DOScheck = 'No'
					else:
						# calculate DOS1 corrected reflectance
						# reflectance = DN / quantificationValue = DN * radC
						# radiance = reflectance * ESUNλ * cosθs / (π * d^2) = DN * radC  * ESUNλ * cosθs / (π * d^2)
						# path radiance Lp = (LDNm * radC - 0.01) * ESUNλ * cosθs / (π * d^2)
						# land surface reflectance ρ = [π * (Lλ - Lp) * d^2]/ (ESUNλ * cosθs) = DN * radC - (LDNm * radC - 0.01)
						e = 'cfg.np.clip(cfg.np.where(raster == ' + str(nD) + ', ' + str(cfg.NoDataVal) + ', ( raster * ' + str("%.16f" % radC) + ' - (- 0.01 + ' + str('%.16f' % radC) + ' * LDNm)) ), 0, 1)'
						functionList.append(e)
						variableList.append(["raster"])
						DOScheck = 'Yes'
					resample = ''
					resampleList.append(resample)
					bandNumberD = {'01': 1, '02': 2, '03': 3, '04': 4, '05': 5, '06': 6, '07': 7, '07': 7, '08': 8, '8a': 9, '09': 10, '10': 11, '11': 12, '12': 13}
					bandSetList.append(str(bandNumberD[str(bNmb).lower()]))
					bandSetNameList.append(oNm)
					inputList.append(inputRaster)
					outputRasterList.append(outputRaster)
		try:
			tPMD = cfg.utls.createTempVirtualRaster(inputList, 'No', 'Yes', 'Yes', 0, 'No', 'No')
		except Exception as err:
			# logger
			cfg.utls.logCondition(str(__name__) + '-' + str(cfg.inspectSCP.stack()[0][3])+ ' ' + cfg.utls.lineOfCode(), ' ERROR exception: ' + str(err))
			if batch == 'No':
				cfg.uiUtls.removeProgressBar()
				cfg.cnvs.setRenderFlag(True)
			cfg.mx.msgErr37()
			return 'No'
		# conversion
		ck = self.sentinelReflectance(tPMD, outputRasterList, functionList, variableList, resampleList, nD, DOScheck)
		cfg.uiUtls.updateBar(90)
		if cfg.actionCheck == 'Yes':
			# load raster bands
			check = 'Yes'
			for outR in outputRasterList:
				if cfg.osSCP.path.isfile(outR):
					cfg.utls.addRasterLayer(outR)
				else:
					cfg.mx.msgErr38(outR)
					cfg.utls.logCondition(str(__name__) + "-" + (cfg.inspectSCP.stack()[0][3])+ " " + cfg.utls.lineOfCode(), "WARNING: unable to load raster" + str(outR))
					check = 'No'
			# create band set
			if cfg.ui.S2_create_bandset_checkBox.isChecked() is True and check == 'Yes':
				if str(sat).lower() in ['sentinel_2a', 'sentinel-2a', 'sentinel_2b', 'sentinel-2b']:
					satName = cfg.satSentinel2
				else:
					satName = 'No'
				if satName != 'No':
					if cfg.ui.add_new_bandset_checkBox_2.isChecked() is True:
						bandSetNumber = cfg.bst.addBandSetTab()
					cfg.bst.rasterBandName()
					bandSetNameList = sorted(bandSetNameList)
					bandSetNameList = [list for (number, list) in sorted(zip(bandSetList, bandSetNameList))]
					cfg.bst.setBandSet(bandSetNameList, bandSetNumber, dt)
					cfg.bandSetsList[bandSetNumber][0] = 'Yes'
					bandSetList = sorted(bandSetList)
					cfg.bst.setSatelliteWavelength(satName, bandSetList, bandSetNumber)
				cfg.bst.bandSetTools(out)
		cfg.uiUtls.updateBar(100)
		if batch == 'No':
			cfg.utls.finishSound()
			cfg.utls.sendSMTPMessage(None, str(__name__))
			cfg.cnvs.setRenderFlag(True)
			cfg.uiUtls.removeProgressBar()

	# calculate DOS1 reflectance
	def sentinelReflectance(self, inputRaster, outputRasterList, functionList, variableList, resampleList, NoData, DOS1Check = 'No'):
		oM = cfg.utls.createTempRasterList(len(outputRasterList))
		try:
			# open input with GDAL
			rD = cfg.gdalSCP.Open(inputRaster, cfg.gdalSCP.GA_ReadOnly)
			# output rasters
			cfg.utls.createRasterFromReference(rD, 1, oM, cfg.NoDataVal, 'GTiff', cfg.rasterDataType, 0,  None, compress = 'Yes', compressFormat = 'LZW')
			rD = None
			if DOS1Check == 'No':
				argumentList = functionList
			else:
				# find DNmin
				LDNmList = cfg.utls.findDNmin(inputRaster, NoData)
				argumentList = []
				for t in range(0, len(outputRasterList)):
					e = functionList[t].replace('LDNm', str(LDNmList[t]))
					argumentList.append(e)
			o = cfg.utls.multiProcessRaster(rasterPath = inputRaster, functionBand = 'No', functionRaster = cfg.utls.calculateRaster, outputRasterList = oM, nodataValue = NoData, functionBandArgument = argumentList, functionVariable = variableList, progressMessage = cfg.QtWidgetsSCP.QApplication.translate('semiautomaticclassificationplugin', 'Conversion'), parallel = cfg.parallelRaster)
			if cfg.actionCheck == 'Yes':
				for t in range(0, len(outputRasterList)):
					cfg.shutilSCP.move(oM[t], outputRasterList[t])
		except Exception as err:
			# logger
			cfg.utls.logCondition(str(__name__) + '-' + str(cfg.inspectSCP.stack()[0][3])+ ' ' + cfg.utls.lineOfCode(), " ERROR exception: " + str(err))
			return 'No'
		# remove temporary files
		try:
			for t in oM:
				cfg.osSCP.remove(t)
		except:
			pass
		# logger
		cfg.utls.logCondition(str(__name__) + '-' + str(cfg.inspectSCP.stack()[0][3])+ ' ' + cfg.utls.lineOfCode(), str(inputRaster))
		return 'Yes'
				