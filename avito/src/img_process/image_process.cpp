// This sample needs at least CUDA 5.5 and a GPU that has at least Compute Capability 2.0

#include <npp.h>
#include <cuda_runtime.h>

#include "Endianess.h"
#include <math.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include "helper_string.h"
#include "helper_cuda.h"
#include "helper_npp.h"

using namespace std ;

struct FrameHeader
{
   unsigned char nSamplePrecision ;
   unsigned short nHeight ;
   unsigned short nWidth ;
   unsigned char nComponents ;
   unsigned char aComponentIdentifier[ 3 ] ;
   unsigned char aSamplingFactors[ 3 ] ;
   unsigned char aQuantizationTableSelector[ 3 ] ;
} ;

struct ScanHeader
{
   unsigned char nComponents ;
   unsigned char aComponentSelector[ 3 ] ;
   unsigned char aHuffmanTablesSelector[ 3 ] ;
   unsigned char nSs ;
   unsigned char nSe ;
   unsigned char nA ;
} ;

struct QuantizationTable
{
   unsigned char nPrecisionAndIdentifier ;
   unsigned char aTable[ 64 ] ;
} ;

struct HuffmanTable
{
   unsigned char nClassAndIdentifier ;
   unsigned char aCodes[ 16 ] ;
   unsigned char aTable[ 256 ] ;
} ;


int divUp( int x, int d )
{
   return ( x + d - 1 ) / d ;
}

template< typename T >
T readAndAdvance( const unsigned char *&pData )
{
   T nElement = readBigEndian<T>( pData ) ;
   pData += sizeof( T ) ;
   return nElement ;
}

int nextMarker( const unsigned char *pData, int &nPos, int nLength )
{
   unsigned char c = pData[ nPos++ ] ;

   do
   {
      while ( c != 0xffu && nPos < nLength )
      {
         c = pData[ nPos++ ] ;
      }

      if ( nPos >= nLength )
         return -1 ;

      c = pData[ nPos++ ] ;
   }
   while ( c == 0 || c == 0x0ffu ) ;

   return c ;
}

void loadJpeg( const char *pFileName, unsigned char *&pJpegData, int &nInputLength )
{
   // Load file into CPU memory
   ifstream stream( pFileName, ifstream::binary ) ;

   if ( !stream.good() )
   {
      return ;
   }

   stream.seekg( 0, ios::end ) ;
   nInputLength = (int)stream.tellg() ;
   stream.seekg( 0, ios::beg ) ;

   pJpegData = new unsigned char[ nInputLength ] ;
   stream.read( reinterpret_cast<char *>( pJpegData ), nInputLength ) ;
}

void readFrameHeader( const unsigned char *pData, FrameHeader &header )
{
   readAndAdvance<unsigned short>( pData ) ;
   header.nSamplePrecision = readAndAdvance<unsigned char>( pData ) ;
   header.nHeight = readAndAdvance<unsigned short>( pData ) ;
   header.nWidth = readAndAdvance<unsigned short>( pData ) ;
   header.nComponents = readAndAdvance<unsigned char>( pData ) ;

   for ( int c = 0 ; c<header.nComponents ; ++c )
   {
      header.aComponentIdentifier[ c ] = readAndAdvance<unsigned char>( pData ) ;
      header.aSamplingFactors[ c ] = readAndAdvance<unsigned char>( pData ) ;
      header.aQuantizationTableSelector[ c ] = readAndAdvance<unsigned char>( pData ) ;
   }
}

void readScanHeader( const unsigned char *pData, ScanHeader &header )
{
   readAndAdvance<unsigned short>( pData ) ;

   header.nComponents = readAndAdvance<unsigned char>( pData ) ;

   for ( int c = 0 ; c<header.nComponents ; ++c )
   {
      header.aComponentSelector[ c ] = readAndAdvance<unsigned char>( pData ) ;
      header.aHuffmanTablesSelector[ c ] = readAndAdvance<unsigned char>( pData ) ;
   }

   header.nSs = readAndAdvance<unsigned char>( pData ) ;
   header.nSe = readAndAdvance<unsigned char>( pData ) ;
   header.nA = readAndAdvance<unsigned char>( pData ) ;
}

void readQuantizationTables( const unsigned char *pData, QuantizationTable *pTables )
{
   unsigned short nLength = readAndAdvance<unsigned short>( pData ) -2 ;

   while ( nLength > 0 )
   {
      unsigned char nPrecisionAndIdentifier = readAndAdvance<unsigned char>( pData ) ;
      int nIdentifier = nPrecisionAndIdentifier & 0x0f ;

      pTables[ nIdentifier ].nPrecisionAndIdentifier = nPrecisionAndIdentifier ;
      memcpy( pTables[ nIdentifier ].aTable, pData, 64 ) ;
      pData += 64 ;

      nLength -= 65 ;
   }
}

void readHuffmanTables( const unsigned char *pData, HuffmanTable *pTables )
{
   unsigned short nLength = readAndAdvance<unsigned short>( pData ) -2 ;

   while ( nLength > 0 )
   {
      unsigned char nClassAndIdentifier = readAndAdvance<unsigned char>( pData ) ;
      int nClass = nClassAndIdentifier >> 4 ; // AC or DC
      int nIdentifier = nClassAndIdentifier & 0x0f ;
      int nIdx = nClass * 2 + nIdentifier ;
      pTables[ nIdx ].nClassAndIdentifier = nClassAndIdentifier ;

      // Number of Codes for Bit Lengths [1..16]
      int nCodeCount = 0 ;

      for ( int i = 0 ; i < 16 ; ++i )
      {
         pTables[ nIdx ].aCodes[ i ] = readAndAdvance<unsigned char>( pData ) ;
         nCodeCount += pTables[ nIdx ].aCodes[ i ] ;
      }

      memcpy( pTables[ nIdx ].aTable, pData, nCodeCount ) ;
      pData += nCodeCount ;

      nLength -= 17 + nCodeCount ;
   }
}

void readRestartInterval( const unsigned char *pData, int &nRestartInterval )
{
   readAndAdvance<unsigned short>( pData ) ;
   nRestartInterval = readAndAdvance<unsigned short>( pData ) ;
}

bool printfNPPinfo( int argc, char *argv[], int cudaVerMajor, int cudaVerMinor )
{
   const NppLibraryVersion *libVer = nppGetLibVersion() ;

   printf( "NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build ) ;

   int driverVersion, runtimeVersion ;
   cudaDriverGetVersion( &driverVersion ) ;
   cudaRuntimeGetVersion( &runtimeVersion ) ;

   printf( "  CUDA Driver  Version: %d.%d\n", driverVersion / 1000, ( driverVersion % 100 ) / 10 ) ;
   printf( "  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, ( runtimeVersion % 100 ) / 10 ) ;

   bool bVal = checkCudaCapabilities( cudaVerMajor, cudaVerMinor ) ;
   return bVal ;
}

// Decode image from jpeg into RGB vectors
Npp8u **decodeImage( const char *inputFile, int &inputStep, NppiSize &inputSize, bool &gray )
{
   unsigned char *pJpegData = 0 ;
   int nInputLength = 0 ;

   // Load Jpeg
   loadJpeg( inputFile, pJpegData, nInputLength ) ;
   if ( pJpegData == 0 )
   {
      cerr << "Input File Error: " << inputFile << endl ;
      return NULL ;
   }

   /***************************
   *
   *   Input
   *
   ***************************/

   // Check if this is a valid JPEG file
   int nPos = 0 ;
   int nMarker = nextMarker( pJpegData, nPos, nInputLength ) ;

   if ( nMarker != 0x0D8 )
   {
      cerr << "Invalid Jpeg Image" << endl ;
      delete[] pJpegData ;
      return NULL ;
   }

   nMarker = nextMarker( pJpegData, nPos, nInputLength ) ;

   // Parsing and Huffman Decoding (on host)
   FrameHeader oFrameHeader ;
   QuantizationTable aQuantizationTables[ 4 ] ;

   HuffmanTable aHuffmanTables[ 4 ] ;
   HuffmanTable *pHuffmanDCTables = aHuffmanTables ;
   HuffmanTable *pHuffmanACTables = &aHuffmanTables[ 2 ] ;
   ScanHeader oScanHeader ;
   memset( &oFrameHeader, 0, sizeof( FrameHeader ) ) ;
   memset( aQuantizationTables, 0, 4 * sizeof( QuantizationTable ) ) ;
   memset( aHuffmanTables, 0, 4 * sizeof( HuffmanTable ) ) ;
   int nMCUBlocksH = 0 ;
   int nMCUBlocksV = 0 ;

   int nRestartInterval = -1 ;

   NppiSize aSrcSize[ 3 ] ;
   Npp16s *aphDCT[ 3 ] = { 0, 0, 0 } ;
   Npp16s *apdDCT[ 3 ] = { 0, 0, 0 } ;
   Npp32s aDCTStep[ 3 ] ;

   Npp8u **apSrcImage = new Npp8u *[ 3 ] ;
   Npp32s aSrcImageStep[ 3 ] ;

   while ( nMarker != -1 )
   {
      if ( nMarker == 0x0D8 )
      {
         // Embedded Thumbnail, skip it
         int nNextMarker = nextMarker( pJpegData, nPos, nInputLength ) ;

         while ( nNextMarker != -1 && nNextMarker != 0x0D9 )
         {
            nNextMarker = nextMarker( pJpegData, nPos, nInputLength ) ;
         }
      }

      if ( nMarker == 0x0DD )
      {
         readRestartInterval( pJpegData + nPos, nRestartInterval ) ;
      }

      if ( ( nMarker == 0x0C0 ) | ( nMarker == 0x0C2 ) )
      {
         //Assert Baseline for this Sample
         //Note: NPP does support progressive jpegs for both encode and decode
         if ( nMarker != 0x0C0 )
         {
            cerr << "The sample does only support baseline JPEG images" << endl ;
            delete[] pJpegData ;
            return NULL ;
         }

         // Baseline or Progressive Frame Header
         readFrameHeader( pJpegData + nPos, oFrameHeader ) ;

         //Assert 3-Channel Image for this Sample
         if ( oFrameHeader.nComponents != 3 )
         {
            cerr << "The sample does only support color JPEG images" << endl ;
            inputSize.width = oFrameHeader.nWidth ;
            inputSize.height = oFrameHeader.nHeight ;
            gray = true ;
            delete[] pJpegData ;
            return NULL ;
         }

         // Compute channel sizes as stored in the JPEG (8x8 blocks & MCU block layout)
         for ( int i = 0 ; i < oFrameHeader.nComponents ; ++i )
         {
            nMCUBlocksV = max( nMCUBlocksV, oFrameHeader.aSamplingFactors[ i ] & 0x0f ) ;
            nMCUBlocksH = max( nMCUBlocksH, oFrameHeader.aSamplingFactors[ i ] >> 4 ) ;
         }

         for ( int i = 0 ; i < oFrameHeader.nComponents ; ++i )
         {
            NppiSize oBlocks ;
            NppiSize oBlocksPerMCU = { oFrameHeader.aSamplingFactors[ i ] >> 4, oFrameHeader.aSamplingFactors[ i ] & 0x0f } ;

            oBlocks.width = (int)ceil( ( oFrameHeader.nWidth + 7 ) / 8 *
                                       static_cast<float>( oBlocksPerMCU.width ) / nMCUBlocksH ) ;
            oBlocks.width = divUp( oBlocks.width, oBlocksPerMCU.width ) * oBlocksPerMCU.width ;

            oBlocks.height = (int)ceil( ( oFrameHeader.nHeight + 7 ) / 8 *
                                        static_cast<float>( oBlocksPerMCU.height ) / nMCUBlocksV ) ;
            oBlocks.height = divUp( oBlocks.height, oBlocksPerMCU.height ) * oBlocksPerMCU.height ;

            aSrcSize[ i ].width = oBlocks.width * 8 ;
            aSrcSize[ i ].height = oBlocks.height * 8 ;

            // Allocate Memory
            size_t nPitch ;
            NPP_CHECK_CUDA( cudaMallocPitch( &apdDCT[ i ], &nPitch, oBlocks.width * 64 * sizeof( Npp16s ), oBlocks.height ) ) ;
            aDCTStep[ i ] = static_cast<Npp32s>( nPitch ) ;

            NPP_CHECK_CUDA( cudaMallocPitch( &apSrcImage[ i ], &nPitch, aSrcSize[ i ].width, aSrcSize[ i ].height ) ) ;
            aSrcImageStep[ i ] = static_cast<Npp32s>( nPitch ) ;

            NPP_CHECK_CUDA( cudaHostAlloc( &aphDCT[ i ], aDCTStep[ i ] * oBlocks.height, cudaHostAllocDefault ) ) ;
         }
      }

      if ( nMarker == 0x0DB )
      {
         // Quantization Tables
         readQuantizationTables( pJpegData + nPos, aQuantizationTables ) ;
      }

      if ( nMarker == 0x0C4 )
      {
         // Huffman Tables
         readHuffmanTables( pJpegData + nPos, aHuffmanTables ) ;
      }

      if ( nMarker == 0x0DA )
      {
         // Scan
         readScanHeader( pJpegData + nPos, oScanHeader ) ;
         nPos += 6 + oScanHeader.nComponents * 2 ;

         int nAfterNextMarkerPos = nPos ;
         int nAfterScanMarker = nextMarker( pJpegData, nAfterNextMarkerPos, nInputLength ) ;

         if ( nRestartInterval > 0 )
         {
            while ( nAfterScanMarker >= 0x0D0 && nAfterScanMarker <= 0x0D7 )
            {
               // This is a restart marker, go on
               nAfterScanMarker = nextMarker( pJpegData, nAfterNextMarkerPos, nInputLength ) ;
            }
         }

         NppiDecodeHuffmanSpec *apHuffmanDCTable[ 3 ] ;
         NppiDecodeHuffmanSpec *apHuffmanACTable[ 3 ] ;

         for ( int i = 0 ; i < 3 ; ++i )
         {
            nppiDecodeHuffmanSpecInitAllocHost_JPEG( pHuffmanDCTables[ ( oScanHeader.aHuffmanTablesSelector[ i ] >> 4 ) ].aCodes, nppiDCTable, &apHuffmanDCTable[ i ] ) ;
            nppiDecodeHuffmanSpecInitAllocHost_JPEG( pHuffmanACTables[ ( oScanHeader.aHuffmanTablesSelector[ i ] & 0x0f ) ].aCodes, nppiACTable, &apHuffmanACTable[ i ] ) ;
         }

         NPP_CHECK_NPP( nppiDecodeHuffmanScanHost_JPEG_8u16s_P3R( pJpegData + nPos, nAfterNextMarkerPos - nPos - 2,
            nRestartInterval, oScanHeader.nSs, oScanHeader.nSe, oScanHeader.nA >> 4, oScanHeader.nA & 0x0f,
            aphDCT, aDCTStep,
            apHuffmanDCTable,
            apHuffmanACTable,
            aSrcSize ) ) ;

         for ( int i = 0 ; i < 3 ; ++i )
         {
            nppiDecodeHuffmanSpecFreeHost_JPEG( apHuffmanDCTable[ i ] ) ;
            nppiDecodeHuffmanSpecFreeHost_JPEG( apHuffmanACTable[ i ] ) ;
         }
      }

      nMarker = nextMarker( pJpegData, nPos, nInputLength ) ;
   }

   NppiDCTState *pDCTState ;
   NPP_CHECK_NPP( nppiDCTInitAlloc( &pDCTState ) ) ;
   Npp8u *pdQuantizationTables ;
   cudaMalloc( &pdQuantizationTables, 64 * 4 ) ;

   // Copy DCT coefficients and Quantization Tables from host to device
   for ( int i = 0 ; i < 4 ; ++i )
   {
      NPP_CHECK_CUDA( cudaMemcpyAsync( pdQuantizationTables + i * 64, aQuantizationTables[ i ].aTable, 64, cudaMemcpyHostToDevice ) ) ;
   }

   for ( int i = 0 ; i < 3 ; ++i )
   {
      NPP_CHECK_CUDA( cudaMemcpyAsync( apdDCT[ i ], aphDCT[ i ], aDCTStep[ i ] * aSrcSize[ i ].height / 8, cudaMemcpyHostToDevice ) ) ;
   }

   // Inverse DCT
   for ( int i = 0 ; i < 3 ; ++i )
   {
      NPP_CHECK_NPP( nppiDCTQuantInv8x8LS_JPEG_16s8u_C1R_NEW( apdDCT[ i ], aDCTStep[ i ],
         apSrcImage[ i ], aSrcImageStep[ i ],
         pdQuantizationTables + oFrameHeader.aQuantizationTableSelector[ i ] * 64,
         aSrcSize[ i ],
         pDCTState ) ) ;
   }

#ifdef _DEBUG
   fprintf( stderr, "nWidth %d nHeight %d\n", oFrameHeader.nWidth, oFrameHeader.nHeight ) ;
   fprintf( stderr, "sample %d %d %d\n", oFrameHeader.aSamplingFactors[ 0 ], oFrameHeader.aSamplingFactors[ 1 ], oFrameHeader.aSamplingFactors[ 2 ] ) ;
   fprintf( stderr, "component %d %d %d\n", oFrameHeader.aComponentIdentifier[ 0 ], oFrameHeader.aComponentIdentifier[ 1 ], oFrameHeader.aComponentIdentifier[ 2 ] ) ;
   fprintf( stderr, "quantization %d %d %d\n", oFrameHeader.aQuantizationTableSelector[ 0 ], oFrameHeader.aQuantizationTableSelector[ 1 ], oFrameHeader.aQuantizationTableSelector[ 2 ] ) ;
   fprintf( stderr, "n component %d n sample precision %d\n", oFrameHeader.nComponents, oFrameHeader.nSamplePrecision ) ;

   fprintf( stderr, "n component %d\n", oScanHeader.nComponents ) ;
   fprintf( stderr, "component selector %d %d %d\n", oScanHeader.aComponentSelector[ 0 ], oScanHeader.aComponentSelector[ 1 ], oScanHeader.aComponentSelector[ 2 ] ) ;

   fprintf( stderr, "huffman selector %d %d %d\n", oScanHeader.aHuffmanTablesSelector[ 0 ], oScanHeader.aHuffmanTablesSelector[ 1 ], oScanHeader.aHuffmanTablesSelector[ 2 ] ) ;
   fprintf( stderr, "nSs %d nSe %d nA %d\n", oScanHeader.nSs, oScanHeader.nSe, oScanHeader.nA ) ;

   fprintf( stderr, "quantization table %d %d %d %d\n", aQuantizationTables[ 0 ].nPrecisionAndIdentifier, aQuantizationTables[ 1 ].nPrecisionAndIdentifier, aQuantizationTables[ 2 ].nPrecisionAndIdentifier, aQuantizationTables[ 3 ].nPrecisionAndIdentifier ) ;
#endif

   delete[] pJpegData ;
   cudaFree( pdQuantizationTables ) ;
   nppiDCTFree( pDCTState ) ;
   for ( int i = 0 ; i < 3 ; ++i )
   {
      cudaFree( apdDCT[ i ] ) ;
      cudaFreeHost( aphDCT[ i ] ) ;
   }

   inputStep = aSrcImageStep[ 0 ] ;
   inputSize.width = oFrameHeader.nWidth ;
   inputSize.height = oFrameHeader.nHeight ;

   return apSrcImage ;
}

int main( int argc, char **argv )
{
   // Min spec is SM 2.0 devices
   if ( printfNPPinfo( argc, argv, 2, 0 ) == false )
   {
      cerr << "jpegNPP requires a GPU with Compute Capability 2.0 or higher" << endl ;
      cudaDeviceReset() ;
      return EXIT_SUCCESS ;
   }

   int resizeStep ;
   NppiSize resizeSize ;
   resizeSize.width = 17 ;
   resizeSize.height = 16 ;
   Npp8u *pResize = nppiMalloc_8u_C1( resizeSize.width, resizeSize.height, &resizeStep ) ;
   unsigned char *pHost = new unsigned char[ resizeSize.width * resizeSize.height ] ;

   // 100 sets of images to be processed
   for ( int set = 0 ; set < 10 ; set++ )
   {
      for ( int subSet = 0 ; subSet < 10 ; subSet++ )
      {
         char outputFile[ 1024 ] ;
         sprintf( outputFile, "img_%d.csv", set * 10 + subSet ) ;
         FILE *pOutputFile = fopen( outputFile, "w+" ) ;
         for ( int imageID = set * 10 + subSet ; imageID < 14666499 ; imageID += 100 )
         {
            if ( imageID % 100000 == 0 ) fprintf( stderr, "%d\n", imageID ) ;
            char szInputFile[ 1024 ] ;
            sprintf( szInputFile, "C:\\Images_%d\\%d\\%d.jpg", set, set * 10 + subSet, imageID ) ;
            int inputStep ;
            NppiSize inputSize ;
            bool gray = false ;
            Npp8u **pInputImage = decodeImage( szInputFile, inputStep, inputSize, gray ) ;
            if ( pInputImage == NULL )
            {
               if ( gray )
               {
                  // Number of color space is smaller than 3
                  fprintf( pOutputFile, "%d,%d,%d,true,", imageID, inputSize.width, inputSize.height ) ;
                  for ( int i = 0 ; i < 8 ; i++ )
                  {
                     fprintf( pOutputFile, "%08x", int( 0 ) ) ;
                  }
                  fprintf( pOutputFile, "\n" ) ;
               }
               continue ;
            }
            if ( inputSize.width <= 17 || inputSize.height <= 16 )
            {
               // Too small images to be resized
               fprintf( pOutputFile, "%d,%d,%d,false,", imageID, inputSize.width, inputSize.height ) ;
               for ( int i = 0 ; i < 8 ; i++ )
               {
                  fprintf( pOutputFile, "%08x", int( 0 ) ) ;
               }
               fprintf( pOutputFile, "\n" ) ;

               for ( int i = 0 ; i < 3 ; ++i )
               {
                  cudaFree( pInputImage[ i ] ) ;
               }
               continue ;
            }
            // copy to packed
            int tempStep ;
            NppiSize tempSize ;
            tempSize.width = inputSize.width ;
            tempSize.height = inputSize.height ;
            Npp8u *pTemp = nppiMalloc_8u_C3( tempSize.width, tempSize.height, &tempStep ) ;

            //fprintf(stderr, "temp step %d size %d x %d\n", tempStep, tempSize.width, tempSize.height) ;
            NPP_CHECK_NPP( nppiCopy_8u_P3C3R( pInputImage, inputStep, pTemp, tempStep, tempSize ) ) ;

            // RGBToGray
            int grayStep ;
            NppiSize graySize ;
            graySize.width = tempSize.width ;
            graySize.height = tempSize.height ;
            Npp8u *pGray = nppiMalloc_8u_C1( graySize.width, graySize.height, &grayStep ) ;
            //fprintf(stderr, "gray step %d size %d x %d\n", grayStep, graySize.width, graySize.height) ;
            NPP_CHECK_NPP( nppiRGBToGray_8u_C3C1R( pTemp, tempStep, pGray, grayStep, graySize ) ) ;

            // Resize
            float scaleFactorWidth = (float)resizeSize.width / (float)graySize.width ;
            float scaleFactorHeight = (float)resizeSize.height / (float)graySize.height ;
            NppiRect oSrcImageROI = { 0, 0, graySize.width, graySize.height } ;
            NppiRect oDstImageROI = { 0, 0, resizeSize.width, resizeSize.height } ;
            NppiInterpolationMode eInterploationMode = NPPI_INTER_SUPER ;
            NPP_CHECK_NPP( nppiResizeSqrPixel_8u_C1R( pGray, graySize, grayStep, oSrcImageROI,
               pResize, resizeStep, oDstImageROI,
               scaleFactorWidth,
               scaleFactorHeight,
               0.0, 0.0, eInterploationMode ) ) ;

            // Download
            NPP_CHECK_CUDA( cudaMemcpy2D( pHost, resizeSize.width, pResize, resizeStep, resizeSize.width, resizeSize.height, cudaMemcpyDeviceToHost ) ) ;

            // Calculate image hash
            int index = 0 ;
            unsigned int decimal[ 8 ] = { 0, 0, 0, 0, 0, 0, 0, 0 } ;
            for ( int row = 0 ; row < resizeSize.height ; row++ )
            {
               for ( int col = 0 ; col < resizeSize.width - 1 ; col++ )
               {
                  unsigned char pixelLeft = pHost[ row * resizeSize.width + col ] ;
                  unsigned char pixelRight = pHost[ row * resizeSize.width + col + 1 ] ;
                  if ( pixelLeft > pixelRight )
                  {
                     decimal[ index / 32 ] += ( 1 << ( index % 32 ) ) ;
                  }
                  index++ ;
               }
            }

            // Output the result
            fprintf( pOutputFile, "%d,%d,%d,false,", imageID, inputSize.width, inputSize.height ) ;
            for ( int i = 0 ; i < 8 ; i++ )
            {
               fprintf( pOutputFile, "%08x", decimal[ i ] ) ;
            }
            fprintf( pOutputFile, "\n" ) ;

            for ( int i = 0 ; i < 3 ; ++i )
            {
               cudaFree( pInputImage[ i ] ) ;
            }

            cudaFree( pTemp ) ;
            cudaFree( pGray ) ;
         }
         fclose( pOutputFile ) ;
      }
   }
   cudaFree( pResize ) ;
   delete[] pHost ;

   cudaDeviceReset() ;
   return EXIT_SUCCESS ;
}
