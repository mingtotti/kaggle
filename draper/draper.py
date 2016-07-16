# -*- coding: utf-8 -*-

import numpy
import pandas
import cv2
import glob
import logging
import os
import sys
import math

def _imgPreprocess ( imgFile, pcntDownsize ) :
   # Read image1
   image = cv2.imread( imgFile )

   # Perform the resizing of the image by pcntDownsize and create a Grayscale
   # version
   dim = ( int( image.shape[ 1 ] * pcntDownsize ),
           int( image.shape[ 0 ] * pcntDownsize ) )
   img = cv2.resize( image, dim, interpolation = cv2.INTER_AREA )
   imgGray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )

   return imgGray

def _imgStitcher ( imgX, imgY, pcntDownsize = 1.0, tryReverse = False ) :
   imgGrayX = _imgPreprocess( imgX, pcntDownsize )
   imgGrayY = _imgPreprocess( imgY, pcntDownsize )

   # Use BRISK to create key points in each image
   brisk = cv2.BRISK_create()
   kpX, desX = brisk.detectAndCompute( imgGrayX, None )
   kpY, desY = brisk.detectAndCompute( imgGrayY, None )

   # Use BruteForce algorithm to detect matches among image keypoints
   dm = cv2.DescriptorMatcher_create( "BruteForce" )
   matches = dm.knnMatch( desX, desY, 2 )
   filteredMatches = []
   for m in matches :
      if len( m ) == 2 and m[ 0 ].distance < m[ 1 ].distance * 0.75 :
         filteredMatches.append( ( m[ 0 ].trainIdx, m[ 0 ].queryIdx ) )

   kpX = numpy.float32( [ kpX[ m[ 1 ] ].pt for m in filteredMatches ] )
   kpX = kpX.reshape( -1, 1, 2 )
   kpY = numpy.float32( [ kpY[ m[ 0 ] ].pt for m in filteredMatches ] )
   kpY = kpY.reshape( -1, 1, 2 )

   # Calculate homography matrix
   H, mask = cv2.findHomography( kpY, kpX, cv2.RANSAC, 4.0 )

   if H is None and not tryReverse :
      # Try again with 100% scaling
      H = _imgStitcher( imgX, imgY, 1.0, True )
   if H is None :
      H = [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]

   return ( H )

def _readInputs ( fileName ) :
   # Initialize file if does not exist
   if not os.path.exists( fileName ) :
      columnNames = [ "group", "imgX", "imgY",
                      "h0", "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8" ]
      df = pandas.DataFrame( columns = columnNames )
      f = open( fileName, "a+" )
      f.write( ",".join( columnNames ) )
      f.write( "\n" )
      f.flush()
      f.close()
      return ( df )

   # Read from file directly
   df = pandas.read_csv( fileName, index_col = False )
   return ( df )

def _processInputs ( fileSet ) :
   inputDF = pandas.DataFrame( [
         [ f,
           f.split( "/" )[ 1 ].split( "." )[ 0 ].split( "_" )[ 0 ],
           f.split( "/" )[ 1 ].split( "." )[ 0 ].split( "_" )[ 1 ]
         ] for f in glob.glob( "%s_sm/*.jpeg" % fileSet ) ] )

   inputDF.columns = [ "path", "group", "imgID" ]

   fileName = "%s.csv" % fileSet

   df = _readInputs( fileName )

   if len( df.index ) == len( inputDF.index ) * 4 :
      return df

   f = open( fileName, "a+" )
   groups = inputDF.group.unique()
   groups.sort()
   for group in groups :
      logging.info( "processing group %s" % group )
      groupedDF = inputDF[ inputDF.group == group ]
      images = groupedDF.imgID.values
      images.sort()
      for i in images :
         for j in images :
            if i == j :
               continue
            res = df[ ( ( df.group == group ) &
                        ( df.imgX == int( i ) ) &
                        ( df.imgY == int( j ) ) ) ]
            if res.index.size == 1 :
               logging.debug( "skip %s %s %s" % ( group, i, j ) )
               continue
            logging.debug( "generating %s %s %s" % ( group, i, j ) )
            imgX = "%s_sm/%s_%d.jpeg" % ( fileSet, group, int( i ) )
            imgY = "%s_sm/%s_%d.jpeg" % ( fileSet, group, int( j ) )
            H = _imgStitcher( imgX, imgY, 0.4 )
            df = df.append ( { "group" : group,
                               "imgX" : int( i ),
                               "imgY" : int( j ),
                               "h0" : H[ 0 ][ 0 ],
                               "h1" : H[ 0 ][ 1 ],
                               "h2" : H[ 0 ][ 2 ],
                               "h3" : H[ 1 ][ 0 ],
                               "h4" : H[ 1 ][ 1 ],
                               "h5" : H[ 1 ][ 2 ],
                               "h6" : H[ 2 ][ 0 ],
                               "h7" : H[ 2 ][ 1 ],
                               "h8" : H[ 2 ][ 2 ] }, ignore_index = True )
            f.write( ( "%s,%d,%d,"
                       "%.15f,%.15f,%.15f,"
                       "%.15f,%.15f,%.15f,"
                       "%15f,%.15f,%.15f\n" ) % ( 
                       group, int ( i ), int ( j ),
                       H[ 0 ][ 0 ], H[ 0 ][ 1 ], H[ 0 ][ 2 ],
                       H[ 1 ][ 0 ], H[ 1 ][ 1 ], H[ 1 ][ 2 ],
                       H[ 2 ][ 0 ], H[ 2 ][ 1 ], H[ 2 ][ 2 ] ) )
   f.flush ()
   f.close ()
   return ( df )

def _selectTransform ( inputDF, group, imgX, imgY ) :
   return inputDF [ ( ( inputDF.group == group ) &
                      ( inputDF.imgX == imgX ) &
                      ( inputDF.imgY == imgY ) ) ]

def _fixMatrix ( inputDF, group, imgX, imgY, imgZ ) :
   hVector = [ "h0", "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8" ]
   selected = _selectTransform( inputDF, group, imgX, imgY )
   first = _selectTransform( inputDF, group, imgX, imgZ )
   second = _selectTransform( inputDF, group, imgZ, imgY )
   a = numpy.array( first[ hVector ] ).reshape( 3, 3 )
   b = numpy.array( second[ hVector ] ).reshape( 3, 3 )
   a = numpy.matrix( a )
   b = numpy.matrix( b )
   res = a * b
   res = res / res[ 2, 2 ]
   inputDF.loc[ selected.index[ 0 ],
                hVector ] = res.reshape( 1, 9 ).tolist()[ 0 ]
   return inputDF

def _predictOrder ( inputDF ) :
   resultDF = pandas.DataFrame( columns = [ "setId", "day" ] )
   groups = inputDF.group.unique()
   groups.sort()
   for group in groups :
      logging.info( "predicting %s" % group ) ;
      groupedDF = inputDF[ inputDF.group == group ]
      tmp = [ 4, 5, 3, 2, 1 ]
      idx = [ -1, -1, -1, -1, -1 ]

      # Pick the candidates for 1 and 2, which have closer zoom levers
      # pick 2
      maxH = 0.0
      maxIdx = -1
      for imgX in range( 0, 5 ) :
         curH = 0.0
         curIdx = 0
         for imgY in range( 0, 5 ) :
            if imgX == imgY : continue
            curH =( curH +
                     abs( groupedDF.h0.values[ imgX * 4 + curIdx ] ) +
                     abs( groupedDF.h4.values[ imgX * 4 + curIdx ] ) )
            curIdx = curIdx + 1
         if maxIdx == -1 or curH > maxH :
            maxIdx = imgX
            maxH = curH
      if maxIdx != -1 :
         idx[ 1 ] = maxIdx
         old = tmp[ maxIdx ]
         for i in range( 0, 5 ) :
            if tmp[ i ] == 2 :
               tmp[ i ] = old
               break
         tmp[ maxIdx ] = 2

      # pick 1
      maxH = 0.0
      maxIdx = -1
      for imgX in range( 0, 5 ) :
         if tmp[ imgX ] == 2 : continue
         curH = 0.0
         curIdx = 0
         for imgY in range( 0, 5 ) :
            if imgX == imgY : continue
            curH = ( curH +
                     abs( groupedDF.h0.values[ imgX * 4 + curIdx ] ) +
                     abs( groupedDF.h4.values[ imgX * 4 + curIdx ] ) )
            curIdx = curIdx + 1
         if maxIdx == -1 or curH > maxH :
            maxIdx = imgX
            maxH = curH
      if maxIdx != -1 :
         idx[ 0 ] = maxIdx
         old = tmp[ maxIdx ]
         for i in range( 0, 5 ) :
            if tmp[ i ] == 1 :
               tmp[ i ] = old
               break
         tmp[ maxIdx ] = 1

      # pick 2 from 2, 1
      maxH = 0.0
      maxIdx = -1
      for imgX in range( 0, 5 ) :
         if tmp[ imgX ] > 2 : continue
         curH = 0.0
         curIdx = 0
         for imgY in range( 0, 5 ) :
            if imgX == imgY : continue
            if tmp[ imgY ] > 2 :
               curIdx = curIdx + 1
               continue
            curH = ( curH +
                     abs( groupedDF.h0.values[ imgX * 4 + curIdx ] ) +
                     abs( groupedDF.h4.values[ imgX * 4 + curIdx ] ) )
            curIdx = curIdx + 1
         if maxIdx == -1 or curH > maxH :
            maxIdx = imgX
            maxH = curH
      if maxIdx != -1 :
         idx[ 1 ] = maxIdx
         old = tmp[ maxIdx ]
         for i in range( 0, 5 ) :
            if tmp[ i ] == 2 :
               tmp[ i ] = old
               break
         tmp[ maxIdx ] = 2

      # pick 3 from 3, 4, 5
      minH1 = 0.0
      minIdx = -1
      maxH1 = 0.0
      maxIdx = -1
      for imgX in range( 0, 5 ) :
         if tmp[ imgX ] > 2 :
            curH1 = 0.0
            curH4 = 0.0
            curIdx = 0
            for imgY in range( 0, 5 ) :
               if imgX == imgY : continue
               if tmp[ imgY ] > 2 :
                  curH1 = ( curH1 +
                            abs( groupedDF.h1.values[ imgX * 4 + curIdx] ) +
                            abs( groupedDF.h3.values[ imgX * 4 + curIdx] ) )
                  curH4 = ( curH4 +
                            abs( groupedDF.h4.values[ imgX * 4 + curIdx] ) )
               curIdx = curIdx + 1
            if minIdx == -1 or curH1 < minH1 :
               minIdx = imgX
               minH1 = curH1
            if maxIdx == -1 or curH1 > maxH1 :
               maxIdx = imgX
               maxH1 = curH1
      if maxIdx != -1 :
         idx[ 2 ] = maxIdx
         old = tmp[ maxIdx ]
         for i in range( 0, 5 ) :
            if tmp[ i ] == 3 :
               tmp[ i ] = old
               break
         tmp[ maxIdx ] = 3

      # pick 4
      minH4 = 0.0
      minIdx = -1
      maxH4 = 0.0
      maxIdx = -1
      for imgX in range( 0, 5 ) :
         if tmp[ imgX ] > 3:
            curH4 = 0.0
            curIdx = 0
            for imgY in range( 0, 5 ) :
               if imgX == imgY : continue
               if tmp[ imgY ] > 2 :  # 3, 4, 5
                  curH4 = ( curH4 +
                            abs( groupedDF.h4.values[ imgX * 4 + curIdx ] ) )
               curIdx = curIdx + 1
            if minIdx == -1 or curH4 < minH4 :
               minIdx = imgX
               minH4 = curH4
            if maxIdx == -1 or curH4 > maxH4 :
               maxIdx = imgX
               maxH4 = curH4
      if maxIdx != -1 :
         idx[ 3 ] = maxIdx
         old = tmp[ maxIdx ]
         for i in range( 0, 5 ) :
            if tmp[ i ] == 4 :
               tmp[ i ] = old
               break
         tmp[ maxIdx ] = 4

      resultDF = resultDF.append( { "setId" : int( group.split( "set" )[ 1 ] ),
                                    "day" : ( "%d %d %d %d %d" %
                                            ( tmp[ 0 ], tmp[ 1 ], tmp[ 2 ],
                                              tmp[ 3 ], tmp[ 4 ] ) ) },
                                  ignore_index = True )

   return resultDF

if __name__ == "__main__":
   program = os.path.basename( sys.argv[ 0 ] )
   logger = logging.getLogger( program )

   logging.basicConfig( format='%(asctime)s: %(levelname)s: %(message)s' )
   logging.root.setLevel( level = logging.DEBUG )

   testDF = _processInputs( "test" )

   testDF.sort_values( by = [ "group", "imgX", "imgY" ], inplace = True )

   testDF = _fixMatrix( testDF, "set104", 2, 5, 4 )
   testDF = _fixMatrix( testDF, "set104", 5, 2, 4 )

   testDF = _fixMatrix( testDF, "set115", 3, 5, 2 )
   testDF = _fixMatrix( testDF, "set115", 5, 3, 2 )
   testDF = _fixMatrix( testDF, "set115", 3, 4, 2 )
   testDF = _fixMatrix( testDF, "set115", 4, 3, 2 )

   testDF = _fixMatrix( testDF, "set135", 3, 1, 2 )
   testDF = _fixMatrix( testDF, "set135", 1, 3, 2 )

   testDF = _fixMatrix( testDF, "set152", 3, 5, 2 )
   testDF = _fixMatrix( testDF, "set152", 5, 3, 2 )

   # 162

   testDF = _fixMatrix( testDF, "set166", 2, 4, 3 )
   testDF = _fixMatrix( testDF, "set166", 4, 2, 3 )

   testDF = _fixMatrix( testDF, "set169", 2, 5, 3 )
   testDF = _fixMatrix( testDF, "set169", 5, 2, 3 )

   testDF = _fixMatrix( testDF, "set206", 3, 1, 2 )
   testDF = _fixMatrix( testDF, "set206", 1, 3, 2 )

   testDF = _fixMatrix( testDF, "set240", 3, 1, 2 )
   testDF = _fixMatrix( testDF, "set240", 1, 3, 2 )
   testDF = _fixMatrix( testDF, "set240", 3, 4, 2 )
   testDF = _fixMatrix( testDF, "set240", 4, 3, 2 )

   testDF = _fixMatrix( testDF, "set253", 3, 2, 1 )
   testDF = _fixMatrix( testDF, "set253", 2, 3, 1 )

   testDF = _fixMatrix( testDF, "set287", 5, 1, 4 )
   testDF = _fixMatrix( testDF, "set287", 5, 2, 4 )
   testDF = _fixMatrix( testDF, "set287", 2, 5, 4 )
   testDF = _fixMatrix( testDF, "set287", 5, 3, 4 )

   testDF = _fixMatrix( testDF, "set310", 4, 1, 2 )
   testDF = _fixMatrix( testDF, "set310", 1, 4, 2 )

   testDF = _fixMatrix( testDF, "set321", 2, 5, 3 )
   testDF = _fixMatrix( testDF, "set321", 5, 2, 3 )

   testDF = _fixMatrix( testDF, "set54", 3, 2, 1 )
   testDF = _fixMatrix( testDF, "set54", 2, 3, 1 )

   testDF = _fixMatrix( testDF, "set67", 5, 4, 3 )
   testDF = _fixMatrix( testDF, "set67", 4, 5, 3 )

   testDF.to_csv( "test.new.csv", header = True, index = False,
                  float_format = "%.15f" )

   resultDF = _predictOrder( testDF )
   resultDF.sort_values( by = "setId", inplace = True )
   resultDF.to_csv( "result.csv", header = True, index = False,
                    float_format = "%1.f" )

