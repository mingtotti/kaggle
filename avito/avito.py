# -*- coding : utf-8 -*-

import numpy
import dtw
import pandas
import os
import time
import datetime
import xgboost
import zipfile
import json
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
from operator import itemgetter
import sys, logging
from gensim.models.doc2vec import DocvecsArray

def _getDocDTWDist ( x, y, lenX, lenY, lenLimit = 200 ) :
   if lenX == 0 or lenY == 0 :
      if lenX == 0 and lenY == 0 :
         return 0.0
      else :
         return 1.0
   x = numpy.array( [ ord( i ) for i in x[ 0 : lenLimit ] ] )
   x = x.reshape( -1, 1 )
   y = numpy.array( [ ord( i ) for i in y[ 0 : lenLimit ] ] )
   y = y.reshape( -1, 1 )

   distFunc = lambda x, y : 0.0 if x == y else 1.0
   dist, cost, acc, path = dtw.dtw( x, y, dist = distFunc )

   return dist

def _loadItems ( setName ) :
   typesItemInfo = { "itemID" : numpy.dtype( int ),
                     "categoryID" : numpy.dtype( int ),
                     "title" : numpy.dtype( str ),
                     "description" : numpy.dtype( str ),
                     "imagesArray" : numpy.dtype( str ),
                     "attrsJSON" : numpy.dtype( str ),
                     "price" : numpy.dtype( float ),
                     "locationID" : numpy.dtype( int ),
                     "metroID" : numpy.dtype( float ),
                     "lat" : numpy.dtype( float ),
                     "lon" : numpy.dtype( float ) }

   # Add "id" column for easy merge
   logging.info( "Load ItemInfo_%s.csv" % setName )
   items = pandas.read_csv( "ItemInfo_%s.csv" % setName, dtype = typesItemInfo )
   items.title.fillna( "", inplace = True )
   items.description.fillna( "", inplace = True )
   items.attrsJSON.fillna( "", inplace = True )
   items.imagesArray.fillna( "", inplace = True )
   items.fillna( -1, inplace = True )

   # train = train.drop([ "generationMethod" ], axis = 1)

   logging.info( "Add text features..." )
   items[ "lenTitle" ] = items[ "title" ].str.len().astype( numpy.int32 )
   items[ "lenDesc" ] = items[ "description" ].str.len().astype( numpy.int32 )
   items[ "lenAttrs" ] = items[ "attrsJSON" ].str.len().astype( numpy.int32 )

   logging.info( "Merge item %s..." % setName )

   location = pandas.read_csv( "Location.csv" )
   category = pandas.read_csv( "Category.csv" )

   items = pandas.merge( items, category, how = "left", on = "categoryID",
                         left_index = True )
   items = pandas.merge( items, location, how = "left", on = "locationID",
                         left_index = True )

   return items

def _prepareItems ( setName ) :
   items = _loadItems( setName )
   for i in [ "X", "Y" ] :
      tmpItems = items.rename( 
         columns = { "itemID" : "itemID%s" % i,
                     "categoryID" : "categoryID%s" % i,
                     "parentCategoryID" : "parentCategoryID%s" % i,
                     "price" : "price%s" % i,
                     "locationID" : "locationID%s" % i,
                     "regionID" : "regionID%s" % i,
                     "metroID" : "metroID%s" % i,
                     "lat" : "lat%s" % i,
                     "lon" : "lon%s" % i,
                     "title" : "title%s" % i,
                     "description" : "description%s" % i,
                     "attrsJSON" : "attrsJSON%s" % i,
                     "lenTitle" : "lenTitle%s" % i,
                     "lenDesc" : "lenDesc%s" % i,
                     "lenAttrs" : "lenAttrs%s" % i,
                     "imagesArray" : "imagesArray%s" % i } )
      yield tmpItems

def _preparePairs ( setName, stepIdx = 1000 ) :
   itemPairTypes = { "itemIDX" : numpy.dtype( int ),
                     "itemIDY" : numpy.dtype( int ),
                     "isDuplicate" : numpy.dtype( int ),
                     "generationMethod" : numpy.dtype( int ) }

   logging.info( "Load ItemPairs_%s.csv" % setName )
   pairs = pandas.read_csv( "ItemPairs_%s.csv" % setName, dtype = itemPairTypes )
   if "id" not in pairs.columns :
      pairs[ "id" ] = pairs.index

   itemsX = None
   itemsY = None

   for i in range( 0, pairs.index.size / stepIdx + 1 ) :
      fileName = "%s/%s_%d_%d.csv" % ( setName, setName, stepIdx, i )
      logging.info( fileName )
      if os.path.exists( fileName ) :
         continue

      subPairs = pairs[ i * stepIdx : ( i + 1 ) * stepIdx ]

      if itemsX is None or itemsY is None :
         itemsX, itemsY = _prepareItems( setName )

      # Add item X data
      subPairs = pandas.merge( subPairs, itemsX, how = "left", on = "itemIDX",
                               left_index = True )
      # Add item Y data
      subPairs = pandas.merge( subPairs, itemsY, how = "left", on = "itemIDY",
                               left_index = True )
      subPairs.to_csv( fileName, header = True, index = False,
                       encoding = "utf-8" )

   return pairs.index.size

def _fixDataType ( df, columnPrefixes, dtype ) :
   for columnPrefix in columnPrefixes :
      df[ "%sX" % columnPrefix ] = df[ "%sX" % columnPrefix ].astype( dtype )
      df[ "%sY" % columnPrefix ] = df[ "%sY" % columnPrefix ].astype( dtype )
   return df

def _getColumnSame( df, columnPrefixes ) :
   for columnPrefix in columnPrefixes :
      df[ "%sSame" % columnPrefix ] = numpy.equal(
            df[ "%sX" % columnPrefix ],
            df[ "%sY" % columnPrefix ] ).astype( numpy.int32 )
   return df

def _loadSubPairs ( fileName ) :
   pairTypes = { "id" : numpy.dtype( int ),
                 "itemIDX" : numpy.dtype( int ),
                 "itemIDY" : numpy.dtype( int ),
                 "isDuplicate" : numpy.dtype( int ),
                 "generationMethod" : numpy.dtype( int ),
                 "categoryIDX" : numpy.dtype( int ),
                 "titleX" : numpy.dtype( str ),
                 "descriptionX" : numpy.dtype( str ),
                 "imagesArrayX" : numpy.dtype( str ),
                 "attrsJSONX" : numpy.dtype( str ),
                 "priceX" : numpy.dtype( float ),
                 "locationIDX" : numpy.dtype( int ),
                 "metroIDX" : numpy.dtype( float ),
                 "latX" : numpy.dtype( float ),
                 "lonX" : numpy.dtype( float ),
                 "lenTitleX" : numpy.dtype( int ),
                 "lenDescX" : numpy.dtype( int ),
                 "lenAttrsX" : numpy.dtype( int ),
                 "parentCategoryIDX" : numpy.dtype( int ),
                 "regionIDX" : numpy.dtype( int ),
                 "categoryIDY" : numpy.dtype( int ),
                 "titleY" : numpy.dtype( str ),
                 "descriptionY" : numpy.dtype( str ),
                 "imagesArrayY" : numpy.dtype( str ),
                 "attrsJSONY" : numpy.dtype( str ),
                 "priceY" : numpy.dtype( float ),
                 "locationIDY" : numpy.dtype( int ),
                 "metroIDY" : numpy.dtype( float ),
                 "latY" : numpy.dtype( float ),
                 "lonY" : numpy.dtype( float ),
                 "lenTitleY" : numpy.dtype( int ),
                 "lenDescY" : numpy.dtype( int ),
                 "lenAttrsY" : numpy.dtype( int ),
                 "parentCategoryIDY" : numpy.dtype( int ),
                 "regionIDY" : numpy.dtype( int ) }
   logging.info( "Load sub pairs %s" % fileName )
   subPairs = pandas.read_csv( fileName, dtype = pairTypes )

   subPairs.titleX.fillna( "", inplace = True )
   subPairs.descriptionX.fillna( "", inplace = True )
   subPairs.attrsJSONX.fillna( "", inplace = True )
   subPairs.imagesArrayX.fillna( "", inplace = True )
   subPairs.titleY.fillna( "", inplace = True )
   subPairs.descriptionY.fillna( "", inplace = True )
   subPairs.attrsJSONY.fillna( "", inplace = True )
   subPairs.imagesArrayY.fillna( "", inplace = True )

   return subPairs

def _transformPairs ( setName, stepIdx, numPairs ) :
   for i in range( 0, numPairs / stepIdx + 1 ) :
      fileName = "%s/%s_%d_%d.csv" % ( setName, setName, stepIdx, i )
      transFileName = "%s/%s_trans_%d_%d.csv" % (
                       setName, setName, stepIdx, i )
      if os.path.exists( transFileName ) :
         logging.info( "skip %s" % transFileName )
         continue
      start = time.time()
      subPairs = _loadSubPairs( fileName )

      subPairs = subPairs.merge(
            subPairs.apply(
                  lambda x : pandas.Series( {
                        "distTitle" : _getDocDTWDist( x.titleX, x.titleY,
                                                      x.lenTitleX,
                                                      x.lenTitleY ),
                        "distDescription" : _getDocDTWDist( x.descriptionX,
                                                            x.descriptionY,
                                                            x.lenDescX,
                                                            x.lenDescY ),
                        "distAttrsJSON" : _getDocDTWDist( x.attrsJSONX,
                                                           x.attrsJSONY,
                                                           x.lenAttrsX,
                                                           x.lenAttrsY ) } ),
                  axis = 1 ),
            left_index = True, right_index = True )

      subPairs.drop( [ "titleX", "titleY", "descriptionX", "descriptionY",
                       "attrsJSONX", "attrsJSONY", "imagesArrayX",
                       "imagesArrayY" ], axis = 1, inplace = True )

      # Create same arrays
      logging.info( "Generated distances" )

      subPairs[ "priceSame" ] = numpy.equal(
            subPairs[ "priceX" ], subPairs[ "priceY" ] ).astype( numpy.int32 )

      subPairs[ "locationIDSame" ] = numpy.equal(
            subPairs[ "locationIDX" ],
            subPairs[ "locationIDY" ] ).astype( numpy.int32 )
      subPairs[ "categoryIDSame" ] = numpy.equal(
            subPairs[ "categoryIDX" ],
            subPairs[ "categoryIDY" ] ).astype( numpy.int32 )
      subPairs[ "regionIDSame" ] = numpy.equal(
            subPairs[ "regionIDX" ],
            subPairs[ "regionIDY" ] ).astype( numpy.int32 )
      subPairs[ "metroIDSame" ] = numpy.equal(
            subPairs[ "metroIDX" ],
            subPairs[ "metroIDY" ] ).astype( numpy.int32 )
      subPairs[ "latSame" ] = numpy.equal(
            subPairs[ "latX" ],
            subPairs[ "latY" ] ).astype( numpy.int32 )
      subPairs[ "lonSame" ] = numpy.equal(
            subPairs[ "lonX" ],
            subPairs[ "lonY" ] ).astype( numpy.int32 )
      subPairs[ "lenTitleSame" ] = numpy.equal(
            subPairs[ "lenTitleX" ],
            subPairs[ "lenTitleY" ] ).astype( numpy.int32 )
      subPairs[ "lenDescSame" ] = numpy.equal(
            subPairs[ "lenDescX" ],
            subPairs[ "lenDescY" ] ).astype( numpy.int32 )
      subPairs[ "lenAttrsSame" ] = numpy.equal(
            subPairs[ "lenAttrsX" ],
            subPairs[ "lenAttrsY" ] ).astype( numpy.int32 )

      lonX = subPairs[ "lonX" ]
      latX = subPairs[ "latX" ]
      lonY = subPairs[ "lonY" ]
      latY = subPairs[ "latY" ]
      subPairs[ "distLocation" ] = pow( pow( lonX - lonY, 2 ) +
                                   pow( latX - latY, 2 ), 0.5 )

      subPairs.to_csv( transFileName, header = True, index = False,
                       encoding = "utf-8" )

      stop = time.time()
      logging.info( "%s cost %.3f" % ( transFileName, stop - start ) )

def _getImgNum ( imageArray ) :
   if ( len( imageArray ) == 0 ) :
      return ( 0, None )
   images = [ int( i ) for i in imageArray.split( ", " ) ]
   return ( len( images ), images )

def _getImage ( imageID, imageDF ) :
   image_set = imageID % 100
   try :
      x = imageDF[ image_set ].loc[ imageID ]
   except :
      x = pandas.Series( {
            "id" : imageID,
            "width" : 0,
            "height" : 0,
            "grey" : True,
            "hashCode" : ( "00000000000000000000000000000000"
                           "00000000000000000000000000000000" ) } )
      logging.info( "empty id : %d" % imageID )
   return x

def _getImagesInfo ( images, imageDF ) :
   res = []
   for imageID in images :
      res.append( _getImage( imageID, imageDF ) )
   return res

def _getImageStat ( imagesInfo ) :
   numGrey = 0
   numVertical = 0
   numHorizonal = 0
   numSquare = 0
   for x in imagesInfo :
      if x.grey :
         numGrey += 1
      if x.width == x.height :
         numSquare += 1
      elif x.width > x.height :
         numHorizonal += 1
      else :
         numVertical += 1
   return numGrey, numVertical, numHorizonal, numSquare

def _getImageHashDiff ( x, y, hexHammingMap ) :
   if len( x ) != len( y ) :
      return 1.0
   strLen = len( x )
   hashDiff = sum( [
         hexHammingMap[ x[ i ] ][ y[ i ] ] for i in range( 0, strLen ) ] )
   return float( hashDiff ) / float( strLen )

def processImage ( x, imageDF, hexHummingMap ) :
   numImageX, imagesX = _getImgNum( x.imagesArrayX )
   numImageY, imagesY = _getImgNum( x.imagesArrayY )

   numGreyX = 0
   numGreyY = 0
   numVerticalX = 0
   numVerticalY = 0
   numHorizonalX = 0
   numHorizonalY = 0
   numSquareX = 0
   numSquareY = 0

   minHashDiff = 0.0
   avgMinHashDiff = 0.0
   minImageDiff = 0.0
   avgMinImageDiff = 0.0

   if numImageX == 0 :
      if numImageY != 0 :
         imagesInfoY = _getImagesInfo( imagesY, imageDF )
         ( numGreyY, numVerticalY, numHorizonalY,
           numSquareY ) = _getImageStat( imagesInfoY )
         minHashDiff = 1.0
         avgMinHashDiff = 1.0
         minImageDiff = 1.0
         avgMinImageDiff = 1.0
   else :
      if numImageY == 0 :
         imagesInfoX = _getImagesInfo( imagesX, imageDF )
         ( numGreyX, numVerticalX, numHorizonalX,
           numSquareX ) = _getImageStat( imagesInfoX )
         minHashDiff = 1.0
         avgMinHashDiff = 1.0
         minImageDiff = 1.0
         avgMinImageDiff = 1.0

   if numImageX > 0 and numImageY > 0 :
         imagesInfoX = _getImagesInfo( imagesX, imageDF )
         ( numGreyX, numVerticalX, numHorizonalX,
           numSquareX ) = _getImageStat( imagesInfoX )
         imagesInfoY = _getImagesInfo( imagesY, imageDF )
         ( numGreyY, numVerticalY, numHorizonalY,
           numSquareY ) = _getImageStat( imagesInfoY )

         hashDiffX = numpy.repeat( 1.0, numImageX )
         hashDiffY = numpy.repeat( 1.0, numImageY )
         for i in range( 0, numImageX ) :
            imageX = imagesInfoX[ i ]
            for j in range( 0, numImageY ) :
               imageY = imagesInfoY[ j ]
               if ( imageX.grey == imageY.grey and
                    imageX.width == imageY.width and
                    imageX.height == imageY.height ) :
                  if imageX.grey and imageY.grey :
                     hashDiffX[ i ] = 0.0
                     hashDiffY[ j ] = 0.0
                  else :
                     temp_diff = _getImageHashDiff( imageX.hashCode,
                                                    imageY.hashCode,
                                                    hexHummingMap )
                     if temp_diff < hashDiffX[ i ] :
                        hashDiffX[ i ] = temp_diff
                     if temp_diff < hashDiffY[ j ] :
                        hashDiffY[ j ] = temp_diff
         minHashDiff = min( min( hashDiffX ), min( hashDiffY ) )
         avgMinHashDiff = ( float( sum( hashDiffX ) + sum( hashDiffY ) ) /
                            float( numImageX + numImageY ) )
         minImageDiff = min( min( numpy.ceil( hashDiffX ) ),
                             min( numpy.ceil( hashDiffY ) ) )
         avgMinImageDiff = ( float( sum( numpy.ceil( hashDiffX ) ) +
                                    sum( numpy.ceil( hashDiffY ) ) ) /
                             float( numImageX + numImageY ) )

   rateVerticalX = ( 0.0 if numImageX == 0
                         else float( numVerticalX ) / float( numImageX ) )
   rateVerticalY = ( 0.0 if numImageY == 0
                         else float( numVerticalY ) / float( numImageY ) )
   rateHorizonalX = ( 0.0 if numImageX == 0
                          else float( numHorizonalX ) / float( numImageX ) )
   rateHorizonalY = ( 0.0 if numImageY == 0
                          else float( numHorizonalY ) / float( numImageY ) )
   rateSquareX = ( 0.0 if numImageX == 0
                       else float( numSquareX ) / float( numImageX ) )
   rateSquareY = ( 0.0 if numImageY == 0
                       else float( numSquareY ) / float( numImageY ) )
   return pandas.Series( { "numImageX" : numImageX,
                           "numImageY" : numImageY,
                           "numGreyX" : numGreyX,
                           "numGreyY" : numGreyY,
                           "numVerticalX" : numVerticalX,
                           "numVerticalY" : numVerticalY,
                           "rateVerticalX" : rateVerticalX,
                           "rateVerticalY" : rateVerticalY,
                           "numHorizonalX" : numHorizonalX,
                           "numHorizonalY" : numHorizonalY,
                           "rateHorizonalX" : rateHorizonalX,
                           "rateHorizonalY" : rateHorizonalY,
                           "numSquareX" : numSquareX,
                           "numSquareY" : numSquareY,
                           "rateSquareX" : rateSquareX,
                           "rateSquareY" : rateSquareY,
                           "minHashDiff" : minHashDiff,
                           "avgMinHashDiff" : avgMinHashDiff,
                           "minImageDiff" : minImageDiff,
                           "avgMinImageDiff" : avgMinImageDiff
      } )

def hammingWeight ( n ) :
   count = 0
   if n == 0 :
      return count
   count += 1
   while n & n - 1 :
      n = n & n - 1
      count = count + 1
   return count

def _hammingMap () :
   map = {}
   for i in range( 0, 16 ) :
      iPos = hex( i )[ 2 : ]
      map[ iPos ] = {}
      for j in range( 0, 16 ) :
         jPos = hex( j )[ 2 : ]
         map[ iPos ][ jPos ] = hammingWeight( i ^ j )
   return map

def _loadImages () :
   imageDF = {}
   imageTypes = {
      "id" : numpy.dtype( int ),
      "width" : numpy.dtype( int ),
      "height" : numpy.dtype( int ),
      "grey" : numpy.dtype( bool ),
      "hashCode" : numpy.dtype( str )
   }
   for i in range( 0, 100 ) :
      imageDF[ i ] = pandas.read_csv( "img_%d.csv" % i,
                                      names = [ "id", "width", "height",
                                                "grey", "hashCode" ],
                                      header = None, dtype = imageTypes )
      imageDF[ i ].index = imageDF[ i ][ "id" ]

   return imageDF

def _transfromPairsImg ( setName, stepIdx, numPairs ) :
   hexHammingMap = None
   imageDF = None

   for i in range( 0, numPairs / stepIdx + 1 ) :
      fileName = "%s/%s_%d_%d.csv" % ( setName, setName, stepIdx, i )
      imageFileName = "%s/%s_img_%d_%d.csv" % ( setName, setName, stepIdx, i )
      if os.path.exists( imageFileName ) :
         continue

      logging.info( "start %s" % imageFileName )

      # Prepare
      if imageDF is None :
         imageDF = _loadImages()
      if hexHammingMap is None :
         hexHammingMap = _hammingMap()

      start = time.time()
      subPairs = _loadSubPairs( fileName )

      subPairs = subPairs.merge(
            subPairs.apply( processImage, axis = 1,
                            args = [ imageDF, hexHammingMap ] ),
            left_index = True, right_index = True )
      subPairs = _fixDataType( subPairs,
                               [ "numImage", "numGrey", "numHorizonal",
                                 "numVertical", "numSquare" ],
                               numpy.int32 )
      subPairs = _getColumnSame( subPairs,
                                 [ "numImage", "numGrey", "numHorizonal",
                                   "numVertical", "numSquare", "rateHorizonal",
                                   "rateVertical", "rateSquare" ] )
      subPairs = subPairs[
            [ "id",
              "numImageX", "numImageY", "numImageSame",
              "numGreyX", "numGreyY", "numGreySame",
              "numVerticalX", "numVerticalY", "numVerticalSame",
              "numHorizonalX", "numHorizonalY", "numHorizonalSame",
              "numSquareX", "numSquareY", "numSquareSame",
              "rateVerticalX", "rateVerticalY", "rateVerticalSame",
              "rateHorizonalX", "rateHorizonalY", "rateHorizonalSame",
              "rateSquareX", "rateSquareY", "rateSquareSame",
              "minHashDiff", "avgMinHashDiff",
              "minImageDiff", "avgMinImageDiff" ] ]
      subPairs.to_csv( imageFileName, header = True, index = False,
                       encoding = "utf-8" )
      stop = time.time()
      logging.info( "%s cost %.3f" % ( imageFileName, stop - start ) )

def _getDescInfo ( description ) :
   lenDesc = len( description )
   if len( description ) == 0 :
      return 0, 0, 0, 0.0, 0.0, 0.0, None
   paragraphs = description.split( "\n" )
   numParas = len( paragraphs )
   numWords = 0
   wordLen = 0
   numEmptyParas = 0
   headWords = []
   for paragraph in paragraphs :
      if len( paragraph.strip() ) == 0 :
         numEmptyParas += 1
         continue
      words = paragraph.split( " " )
      wordLen += sum( [ len( word ) for word in words ] )
      paraNumWords = len( words )
      numWords += paraNumWords
      headWords.append( words[ 0 ] )
   avgWordLen = wordLen / numWords
   avgParaLen = lenDesc / numParas
   avgParaWords = numWords / numParas

   return numParas, numWords, numEmptyParas, avgParaLen, avgParaWords, avgWordLen, headWords

def _processDescription ( x ) :
   ( numParasX, numWordsX, numEmptyParasX, avgParaLenX, avgParaWordsX,
     avgWordLenX, headWordsX ) = _getDescInfo( x.descriptionX )
   ( numParasY, numWordsY, numEmptyParasY, avgParaLenY, avgParaWordsY,
     avgWordLenY, headWordsY ) = _getDescInfo( x.descriptionY )
   diffHeadWords = 0.0
   avgDiffHeadWords = 0.0
   numHeadWordsX = 0 if headWordsX is None else len( headWordsX )
   numHeadWordsY = 0 if headWordsY is None else len( headWordsY )
   if numHeadWordsX == 0 and numHeadWordsY == 0 :
      diffHeadWords = 1.0
      avgDiffHeadWords = 1.0
   elif numHeadWordsX != 0 and numHeadWordsY != 0 :
      diffHeadWordsX = numpy.repeat( 0.0, numHeadWordsX )
      diffHeadWordsY = numpy.repeat( 0.0, numHeadWordsY )
      for i in range( 0, numHeadWordsX ) :
         wordX = headWordsX[ i ]
         for j in range( 0, numHeadWordsY ) :
            wordY = headWordsY[ j ]
            tempDiff = 0.0
            if wordX == wordY :
               tempDiff = 1.0
            if tempDiff > diffHeadWordsX[ i ] :
               diffHeadWordsX[ i ] = tempDiff
            if tempDiff > diffHeadWordsY[ j ] :
               diffHeadWordsY[ j ] = tempDiff
      diffHeadWords = max( max( diffHeadWordsX ), max( diffHeadWordsY ) )
      avgDiffHeadWords = ( float( sum( diffHeadWordsX ) +
                                  sum( diffHeadWordsY ) ) /
                           float( numHeadWordsX + numHeadWordsY ) )
   return pandas.Series( { "numParasX" : numParasX,
                           "numWordsX" : numWordsX,
                           "numEmptyParasX" : numEmptyParasX,
                           "avgParaLenX" : avgParaLenX,
                           "avgParaWordsX" : avgParaWordsX,
                           "avgWordLenX" : avgWordLenX,
                           "numParasY" : numParasY,
                           "numWordsY" : numWordsY,
                           "numEmptyParasY" : numEmptyParasY,
                           "avgParaLenY" : avgParaLenY,
                           "avgParaWordsY" : avgParaWordsY,
                           "avgWordLenY" : avgWordLenY,
                           "numHeadWordsX" : numHeadWordsX,
                           "numHeadWordsY" : numHeadWordsY,
                           "diffHeadWords" : diffHeadWords,
                           "avgDiffHeadWords" : avgDiffHeadWords } )

def _transfromPairsDescription ( setName, stepIdx, numPairs ) :
   for i in range( 0, numPairs / stepIdx + 1 ) :
      fileName = "%s/%s_%d_%d.csv" % ( setName, setName, stepIdx, i )
      descFileName = "%s/%s_desc_%d_%d.csv" % ( setName, setName, stepIdx, i )
      if os.path.exists( descFileName ) :
         continue
      start = time.time()
      subPairs = _loadSubPairs( fileName )

      subPairs = subPairs.merge( subPairs.apply( _processDescription,
                                                 axis = 1 ),
                                 left_index = True, right_index = True )
      subPairs = _fixDataType( subPairs,
                               [ "numParas", "numWords", "numEmptyParas",
                                "numHeadWords" ],
                               numpy.int32 )
      subPairs = _getColumnSame( subPairs,
                                 [ "numParas", "numWords", "numEmptyParas",
                                   "numHeadWords", "avgParaLen",
                                   "avgParaWords", "avgWordLen" ] )
      subPairs = subPairs[
            [ "id",
              "numParasX", "numParasY", "numParasSame",
              "numWordsX", "numWordsY", "numWordsSame",
              "numEmptyParasX", "numEmptyParasY", "numEmptyParasSame",
              "avgParaLenX", "avgParaLenY", "avgParaLenSame",
              "avgParaWordsX", "avgParaWordsY", "avgParaWordsSame",
              "avgWordLenX", "avgWordLenY", "avgWordLenSame",
              "numHeadWordsX", "numHeadWordsY", "numHeadWordsSame",
              "diffHeadWords", "avgDiffHeadWords" ] ]
      subPairs.to_csv( descFileName, header = True, index = False,
                       encoding = "utf-8" )
      stop = time.time()
      logging.info( "%s cost %.3f" % ( descFileName, stop - start ) )

def _getAttrsInfo ( attrsJSON ) :
   if len( attrsJSON ) == 0 :
      return 0, None
   tmp = attrsJSON.decode( "utf-8" ).lower()
   attrs = json.loads( tmp )
   numAttrs = len( attrs )
   return numAttrs, attrs

def _processAttrs( x ) :
   numAttrsX, attrsX = _getAttrsInfo( x.attrsJSONX )
   numAttrsY, attrsY = _getAttrsInfo( x.attrsJSONY )
   diffAttrs = 0.0
   avgDiffAttrs = 0.0
   if numAttrsX == 0 and numAttrsY == 0 :
      diffAttrs = 1.0
      avgDiffAttrs = 1.0
   elif numAttrsX != 0 and numAttrsY != 0 :
      diffAttrsX = numpy.repeat( 0.0, numAttrsX )
      diffAttrsY = numpy.repeat( 0.0, numAttrsY )
      indexX = 0
      for i in attrsX :
         attrX = attrsX[ i ]
         indexY = 0
         for j in attrsY :
            if i != j :
               indexY += 1
               continue
            attrY = attrsY[ j ]
            tempDiff = 0.0
            if attrX == attrY :
               tempDiff = 1.0
            else :
               tempDiff = 0.0
            if tempDiff > diffAttrsX[ indexX ] :
               diffAttrsX[ indexX ] = tempDiff
            if tempDiff > diffAttrsY[ indexY ] :
               diffAttrsY[ indexY ] = tempDiff
            indexY += 1
         indexX += 1
      diffAttrs = max( max( diffAttrsX ), max( diffAttrsY ) )
      avgDiffAttrs = ( float( sum( diffAttrsX ) + sum( diffAttrsY ) ) /
                       float( numAttrsX + numAttrsY ) )
   return pandas.Series( { "numAttrsX" : numAttrsX,
                           "numAttrsY" : numAttrsY,
                           "diffAttrs" : diffAttrs,
                           "avgDiffAttrs" : avgDiffAttrs } )

def _transformPairsAttrs ( setName, stepIdx, numPairs ) :
   for i in range( 0, numPairs / stepIdx + 1 ) :
      fileName = "%s/%s_%d_%d.csv" % ( setName, setName, stepIdx, i )
      attrsFileName = "%s/%s_attrs_%d_%d.csv" % ( setName, setName,
                                                   stepIdx, i )
      if os.path.exists( attrsFileName ) :
         continue
      start = time.time()
      subPairs = _loadSubPairs( fileName )

      subPairs = subPairs.merge( subPairs.apply( _processAttrs, axis = 1 ),
                                 left_index = True, right_index = True )
      subPairs = _fixDataType( subPairs, [ "numAttrs" ], numpy.int32 )
      subPairs = _getColumnSame( subPairs, [ "numAttrs" ] )
      subPairs = subPairs[ [ "id",
                             "numAttrsX", "numAttrsY", "numAttrsSame",
                             "diffAttrs", "avgDiffAttrs" ] ]
      subPairs.to_csv( attrsFileName, header = True, index = False,
                       encoding = "utf-8" )
      stop = time.time()
      logging.info( "%s cost %.3f" % ( attrsFileName, stop - start ) )

# ## process title
def _getTitleInfo ( title ) :
   if len( title ) == 0 :
      return 0, None
   tmp = title.decode( "utf-8" ).lower()
   words = tmp.split()
   numTitleWords = len( words )
   return numTitleWords, words

def _processTitle ( x, doc2vecModel, setName ) :
   numTitleWordsX, wordsX = _getTitleInfo( x.titleX )
   numTitleWordsY, wordsY = _getTitleInfo( x.titleY )
   diffTitle = 0.0
   avgDiffTitle = 0.0
   simTitle = -1.0
   if numTitleWordsX == 0 and numTitleWordsY == 0 :
      diffTitle = 1.0
      avgDiffTitle = 1.0
   elif numTitleWordsX != 0 and numTitleWordsY != 0 :
      diffTitleX = numpy.repeat( 0.0, numTitleWordsX )
      diffTitleY = numpy.repeat( 0.0, numTitleWordsY )
      for i in range( 0, numTitleWordsX ) :
         wordX = wordsX[ i ]
         for j in range( 0, numTitleWordsY ) :
            wordY = wordsY[ j ]
            tempDiff = 0.0
            if wordX == wordY :
               tempDiff = 1.0
            else :
               tempDiff = 0.0
            if tempDiff > diffTitleX[ i ] :
               diffTitleX[ i ] = tempDiff
            if tempDiff > diffTitleY[ j ] :
               diffTitleY[ j ] = tempDiff
      diffTitle = max( max( diffTitleX ), max( diffTitleY ) )
      avgDiffTitle = ( float( sum( diffTitleX ) + sum( diffTitleY ) ) /
                       float( numTitleWordsX + numTitleWordsY ) )

      titleTagX = "%s_%d_%d" % ( setName, x.id, 1 )
      titleTagY = "%s_%d_%d" % ( setName, x.id, 2 )
      try :
         simTitle = doc2vecModel.similarity( titleTagX, titleTagY )
      except :
         logging.info( "could not find similarity for %d" % x.id )
         simTitle = -1.0
   return pandas.Series( { "numTitleWordsX" : numTitleWordsX,
                           "numTitleWordsY" : numTitleWordsY,
                           "diffTitle" : diffTitle,
                           "avgDiffTitle" : avgDiffTitle,
                           "simTitle" : simTitle } )

def _transfromPairsTitle ( setName, stepIdx, numPairs ) :
   for i in range( 0, numPairs / stepIdx + 1 ) :
      fileName = "%s/%s_%d_%d.csv" % ( setName, setName, stepIdx, i )
      titleFileName = "%s/%s_title_%d_%d.csv" % ( setName, setName,
                                                  stepIdx, i )
      doc2vecModelName = "title_%s_%d.docvecs" % ( setName, i )
      if os.path.exists( titleFileName ) :
         continue
      doc2vecModel = DocvecsArray.load( doc2vecModelName )
      start = time.time()
      subPairs = _loadSubPairs( fileName )

      subPairs = subPairs.merge( subPairs.apply( _processTitle, axis = 1,
                                                 args = [ doc2vecModel,
                                                          setName ] ),
                                 left_index = True, right_index = True )
      subPairs = _fixDataType( subPairs, [ "numTitleWords" ], numpy.int32 )
      subPairs = _getColumnSame( subPairs, [ "numTitleWords" ] )
      subPairs = subPairs[
            [ "id",
              "numTitleWordsX", "numTitleWordsY", "numTitleWordsSame",
              "diffTitle", "avgDiffTitle", "simTitle" ] ]
      subPairs.to_csv( titleFileName, header = True, index = False,
                       encoding = "utf-8" )
      stop = time.time()
      logging.info( "%s cost %.3f" % ( titleFileName, stop - start ) )

def _mergePairs ( setName, stepIdx, numPairs ) :
   pairTypes = { "id" : numpy.dtype( int ),
                 "itemIDX" : numpy.dtype( int ),
                 "itemIDY" : numpy.dtype( int ),
                 "isDuplicate" : numpy.dtype( int ),
                 "generationMethod" : numpy.dtype( int ),
                 "categoryIDX" : numpy.dtype( int ),
                 "priceX" : numpy.dtype( float ),
                 "locationIDX" : numpy.dtype( int ),
                 "metroIDX" : numpy.dtype( float ),
                 "latX" : numpy.dtype( float ),
                 "lonX" : numpy.dtype( float ),
                 "lenTitleX" : numpy.dtype( int ),
                 "lenDescX" : numpy.dtype( int ),
                 "lenAttrsX" : numpy.dtype( int ),
                 "parentCategoryIDX" : numpy.dtype( int ),
                 "regionIDX" : numpy.dtype( int ),
                 "categoryIDY" : numpy.dtype( int ),
                 "priceY" : numpy.dtype( float ),
                 "locationIDY" : numpy.dtype( int ),
                 "metroIDY" : numpy.dtype( float ),
                 "latY" : numpy.dtype( float ),
                 "lonY" : numpy.dtype( float ),
                 "lenTitleY" : numpy.dtype( int ),
                 "lenDescY" : numpy.dtype( int ),
                 "lenAttrsY" : numpy.dtype( int ),
                 "parentCategoryIDY" : numpy.dtype( int ),
                 "regionIDY" : numpy.dtype( int ),
                 "distTitle" : numpy.dtype( float ),
                 "distDescription" : numpy.dtype( float ),
                 "distAttrsJSON" : numpy.dtype( float ),
                 "distLocation" : numpy.dtype( float ),
                 "priceSame" : numpy.dtype( int ),
                 "locationIDSame" : numpy.dtype( int ),
                 "categoryIDSame" : numpy.dtype( int ),
                 "regionIDSame" : numpy.dtype( int ),
                 "metroIDSame" : numpy.dtype( int ),
                 "latSame" : numpy.dtype( int ),
                 "lonSame" : numpy.dtype( int ),
                 "lenTitleSame" : numpy.dtype( int ),
                 "lenDescSame" : numpy.dtype( int ),
                 "lenAttrsSame" : numpy.dtype( int ),
                 "numImageX" : numpy.dtype( int ),
                 "numImageY" : numpy.dtype( int ),
                 "numImageSame" : numpy.dtype( int ),
                 "numGreyX" : numpy.dtype( int ),
                 "numGreyY" : numpy.dtype( int ),
                 "numGreySame" : numpy.dtype( int ),
                 "numVerticalX" : numpy.dtype( int ),
                 "numVerticalY" : numpy.dtype( int ),
                 "numVerticalSame" : numpy.dtype( int ),
                 "numHorizonalX" : numpy.dtype( int ),
                 "numHorizonalY" : numpy.dtype( int ),
                 "numHorizonalSame" : numpy.dtype( int ),
                 "numSquareX" : numpy.dtype( int ),
                 "numSquareY" : numpy.dtype( int ),
                 "numSquareSame" : numpy.dtype( int ),
                 "rateVerticalX" : numpy.dtype( float ),
                 "rateVerticalY" : numpy.dtype( float ),
                 "rateVerticalSame" : numpy.dtype( float ),
                 "rateHorizonalX" : numpy.dtype( float ),
                 "rateHorizonalY" : numpy.dtype( float ),
                 "rateHorizonalSame" : numpy.dtype( float ),
                 "rateSquareX" : numpy.dtype( float ),
                 "rateSquareY" : numpy.dtype( float ),
                 "rateSquareSame" : numpy.dtype( float ),
                 "minHashDiff" : numpy.dtype( float ),
                 "avgMinHashDiff" : numpy.dtype( float ),
                 "minImageDiff" : numpy.dtype( float ),
                 "avgMinImageDiff" : numpy.dtype( float ),
                 "diffPrice" : numpy.dtype( float ),
                 "diffPriceRatioX" : numpy.dtype( float ),
                 "diffPriceRatioY" : numpy.dtype( float ),
                 "numParasX" : numpy.dtype( int ),
                 "numWordsX" : numpy.dtype( int ),
                 "numEmptyParasX" : numpy.dtype( int ),
                 "avgParaLenX" : numpy.dtype( float ),
                 "avgParaWordsX" : numpy.dtype( float ),
                 "avgWordLenX" : numpy.dtype( float ),
                 "numParasY" : numpy.dtype( int ),
                 "numWordsY" : numpy.dtype( int ),
                 "numEmptyParasY" : numpy.dtype( int ),
                 "avgParaLenY" : numpy.dtype( float ),
                 "avgParaWordsY" : numpy.dtype( float ),
                 "avgWordLenY" : numpy.dtype( float ),
                 "numParasSame" : numpy.dtype( int ),
                 "numWordsSame" : numpy.dtype( int ),
                 "numEmptyParasSame" : numpy.dtype( int ),
                 "avgParaLenSame" : numpy.dtype( int ),
                 "avgParaWordsSame" : numpy.dtype( int ),
                 "avgWordLenSame" : numpy.dtype( int ),
                 "numHeadWordsX" : numpy.dtype( int ),
                 "numHeadWordsY" : numpy.dtype( int ),
                 "numHeadWordsSame" : numpy.dtype( int ),
                 "diffHeadWords" : numpy.dtype( float ),
                 "avgDiffHeadWords" : numpy.dtype( float ),
                 "numAttrsX" : numpy.dtype( int ),
                 "numAttrsY" : numpy.dtype( int ),
                 "numAttrsSame" : numpy.dtype( int ),
                 "diffAttrs" : numpy.dtype( float ),
                 "avgDiffAttrs" : numpy.dtype( float ),
                 "numTitleWordsX" : numpy.dtype( int ),
                 "numTitleWordsY" : numpy.dtype( int ),
                 "numTitleWordsSame" : numpy.dtype( int ),
                 "diffTitle" : numpy.dtype( float ),
                 "avgDiffTitle" : numpy.dtype( float ),
                 "simTitle" : numpy.dtype( float ) }
   pairsFileName = "%s/%s_pairs.csv" % ( setName, setName )
   if os.path.exists( pairsFileName ) :
      pairs = pandas.read_csv( pairsFileName, dtype = pairTypes )
      return pairs
   pairs = None
   for i in range( 0, numPairs / stepIdx + 1 ) :
      transFileName = "%s/%s_trans_%d_%d.csv" % ( setName, setName, stepIdx, i )
      if not os.path.exists( transFileName ) :
         logging.info( "file %s not found" % transFileName )
      imageFileName = "%s/%s_img_%d_%d.csv" % ( setName, setName, stepIdx, i )
      if not os.path.exists( imageFileName ) :
         logging.info( "file %s not found" % imageFileName )
      descFileName = "%s/%s_desc_%d_%d.csv" % ( setName, setName, stepIdx, i )
      if not os.path.exists( descFileName ) :
         logging.info( "file %s not found" % descFileName )
      attrsFileName = "%s/%s_attrs_%d_%d.csv" % ( setName, setName, stepIdx, i )
      if not os.path.exists( attrsFileName ) :
         logging.info( "file %s not found" % attrsFileName )
      titleFileName = "%s/%s_title_%d_%d.csv" % ( setName, setName, stepIdx, i )
      if not os.path.exists( titleFileName ) :
         logging.info( "file %s not found" % titleFileName )
      subTrans = pandas.read_csv( transFileName, dtype = pairTypes )
      subImage = pandas.read_csv( imageFileName, dtype = pairTypes )
      subDesc = pandas.read_csv( descFileName, dtype = pairTypes )
      subAttrs = pandas.read_csv( attrsFileName, dtype = pairTypes )
      subTitle = pandas.read_csv( titleFileName, dtype = pairTypes )
      subPairs = subTrans.merge( subImage, left_on = [ "id" ],
                                 right_on = [ "id" ] )
      subPairs = subPairs.merge( subDesc, left_on = [ "id" ],
                                 right_on = [ "id" ] )
      subPairs = subPairs.merge( subAttrs, left_on = [ "id" ],
                                 right_on = [ "id" ] )
      subPairs = subPairs.merge( subTitle, left_on = [ "id" ],
                                 right_on = [ "id" ] )

      if pairs is None :
         pairs = subPairs
      else :
         pairs = pairs.append( subPairs, ignore_index = True )

   pairs[ "diffPrice" ] = numpy.abs( pairs[ "priceX" ] - pairs[ "priceY" ] )
   pairs[ "diffPriceRatioX" ] = pairs[ "diffPrice" ] / pairs[ "priceX" ]
   pairs[ "diffPriceRatioY" ] = pairs[ "diffPrice" ] / pairs[ "priceY" ]

   pairs.to_csv( pairsFileName, header = True, index = False,
                 encoding = "utf-8" )
   return pairs

def _createFeatureMap ( features ) :
   outFile = open( "xgb.fmap", "w" )
   for i, feat in enumerate( features ) :
      outFile.write( "{0}\t{1}\tq\n".format( i, feat ) )
   outFile.close()

def _getImportance ( gbm, features ) :
      _createFeatureMap( features )
      importance = gbm.get_fscore( fmap = "xgb.fmap" )
      importance = sorted( importance.items(), key = itemgetter( 1 ),
                           reverse = True )
      return importance

def _runTest ( train, test, features, target, randomState = 0 ) :
   eta = 0.1
   maxDepth = 11   # 15
   subSample = 0.8
   colSampleBytree = 0.8
   startTime = time.time()

   logging.info( ( "XGBoost params. ETA : {}, MAX_DEPTH : {}, "
                   "SUBSAMPLE : {}, COLSAMPLE_BY_TREE : {}" ).format(
                        eta, maxDepth, subsample, colSampleBytree ) )
   params = { "objective" : "binary :logistic",
              "booster" : "gbtree",
              "eval_metric" : "auc",
              "eta" : eta,
              "max_depth" : maxDepth,
              "subsample" : subSample,
              "colsample_bytree" : colSampleBytree,
              "silent" : 1,
              "seed" : randomState,
              "nthread" : 4 }
   numBoostRound = 250
   earlyStoppingRounds = 20

   dTrain = xgboost.DMatrix( train[ features ], train[ target ] )

   watchList = [ ( dTrain, "train" ) ]
   gbm = xgboost.train( params, dTrain, numBoostRound, evals = watchList,
                        early_stopping_rounds = earlyStoppingRounds,
                        verbose_eval = True )

   logging.info( "Validating..." )
   check = gbm.predict( xgboost.DMatrix( train[ features ] ), ntree_limit = gbm.best_ntree_limit )
   score = roc_auc_score( train[ target ].values, check )
   logging.info( "Check error value : { :.6f}".format( score ) )

   imp = _getImportance( gbm, features )
   print( "Importance array : ", imp )

   logging.info( "Predict test set..." )
   testPrediction = gbm.predict( xgboost.DMatrix( test[ features ] ),
                                 ntree_limit = gbm.best_ntree_limit )

   logging.info( "Training time : {} minutes".format(
         round( ( time.time() - startTime ) / 60, 2 ) ) )
   return testPrediction.tolist(), score

def _createSubmission ( score, test, prediction ) :
   # Make Submission
   now = datetime.datetime.now()
   submitFile = ( "submission_" + str( now.strftime( "%Y-%m-%d-%H-%M" ) ) +
                "_" + str( score ) + ".csv" )
   logging.info( "Writing submission : ", submitFile )
   f = open( submitFile, "w" )
   f.write( "id,probability\n" )
   total = 0
   for id in test[ "id" ] :
      result = str( id ) + "," + str( prediction[ total ] )
      result += "\n"
      total += 1
      f.write( result )
   f.close()

   logging.info( "Creating zip-file..." )
   z = zipfile.ZipFile( submitFile + ".zip", "w", zipfile.ZIP_DEFLATED )
   z.write( submitFile )
   z.close()

if __name__ == "__main__" :
   program = os.path.basename( sys.argv[ 0 ] )
   logger = logging.getLogger( program )

   logging.basicConfig( format = "%(asctime)s : %(levelname)s : %(message)s" )
   logging.root.setLevel( level = logging.INFO )

   stepIdx = 100000
   numTrainPairs = _preparePairs( "train", stepIdx )
   numTestPairs = _preparePairs( "test", stepIdx )

   _transformPairs( "train", stepIdx, numTrainPairs )
   _transformPairs( "test", stepIdx, numTestPairs )

   _transfromPairsImg( "train", stepIdx, numTrainPairs )
   _transfromPairsImg( "test", stepIdx, numTestPairs )
   _transfromPairsDescription( "train", stepIdx, numTrainPairs )
   _transfromPairsDescription( "test", stepIdx, numTestPairs )
   _transformPairsAttrs( "train", stepIdx, numTrainPairs )
   _transformPairsAttrs( "test", stepIdx, numTestPairs )
   _transfromPairsTitle( "train", stepIdx, numTrainPairs )
   _transfromPairsTitle( "test", stepIdx, numTestPairs )

   trainPairs = _mergePairs( "train", stepIdx, numTrainPairs )
   testPairs = _mergePairs( "test", stepIdx, numTestPairs )

   features = [ "categoryIDX", "priceX", "locationIDX", "metroIDX", "latX",
                "lonX", "lenTitleX", "lenDescX", "lenAttrsX",
                "parentCategoryIDX", "regionIDX",
                "categoryIDY", "priceY", "locationIDY", "metroIDY", "latY",
                "lonY", "lenTitleY", "lenDescY", "lenAttrsY",
                "parentCategoryIDY", "regionIDY",
                "distTitle", "distDescription", "distAttrsJSON", "distLocation",
                "priceSame", "locationIDSame", "categoryIDSame",
                "regionIDSame", "metroIDSame", "latSame", "lonSame",
                "lenTitleSame", "lenDescSame", "lenAttrsSame",
                "diffPrice", "diffPriceRatioX", "diffPriceRatioY",
                "numImageX", "numImageY", "numImageSame",
                "numGreyX", "numGreyY", "numGreySame",
                "numVerticalX", "numVerticalY", "numVerticalSame",
                "numHorizonalX", "numHorizonalY", "numHorizonalSame",
                "numSquareX", "numSquareY", "numSquareSame",
                "rateVerticalX", "rateVerticalY", "rateVerticalSame",
                "rateHorizonalX", "rateHorizonalY", "rateHorizonalSame",
                "rateSquareX", "rateSquareY", "rateSquareSame",
                "minHashDiff", "avgMinHashDiff", "minImageDiff",
                "avgMinImageDiff",
                "numParasX", "numWordsX", "numEmptyParasX", "avgParaLenX",
                "avgParaWordsX", "avgWordLenX",
                "numParasY", "numWordsY", "numEmptyParasY", "avgParaLenY",
                "avgParaWordsY", "avgWordLenY",
                "numParasSame", "numWordsSame", "numEmptyParasSame",
                "avgParaLenSame", "avgParaWordsSame", "avgWordLenSame",
                "numHeadWordsX", "numHeadWordsY", "numHeadWordsSame",
                "diffHeadWords", "avgDiffHeadWords",
                "numAttrsX", "numAttrsY", "numAttrsSame",
                "diffAttrs", "avgDiffAttrs",
                "numTitleWordsX", "numTitleWordsY", "numTitleWordsSame",
                "diffTitle", "avgDiffTitle", "simTitle" ]
   target = "isDuplicate"
   logging.info( "Length of train : ", len( trainPairs ) )
   logging.info( "Length of test : ", len( testPairs ) )
   logging.info( "Features [ {} ] : {}".format( len( features ),
                                                sorted( features ) ) )
   testPrediction, score = _runTest( trainPairs, testPairs, features, target )
   logging.info( "Real score = {}".format( score ) )
   _createSubmission( score, testPairs, testPrediction )
