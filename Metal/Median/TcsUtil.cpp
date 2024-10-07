//
//                      T c s  U t i l . c p p
//
//  Function:
//     A collection of assorted utility routines used by the TCS code.
//
//  Description:
//     The TcsUtil class acts as a repository for unrelated utility
//     routines that are used by various bits of the TCS code.
//
//  Author(s): Keith Shortridge, AAO
//
//  History:
//     16th Nov 2005. Original version, KS.
//      5th Dec 2005. Added ExpandFileName(). KS.
//     12th Dec 2005. Added FormatArcsec(). KS.
//      4th May 2006. Added MatchCaseBlind(). KS.
//     16th May 2006. Include of ctype.h added. KS.
//     17th May 2006. Added SetAsRealTimeHighPriority(). KS.
//     22nd May 2006. Added EncodeReason(). KS.
//     29th May 2006. Added ScheduleQuantum(). KS.
//     14th Aug 2006. Moved EncodeReason() to the TCS_DRAMATask code, allowing
//                    this collection of general utilities to be independent
//                    of DRAMA. (Which makes it easier to use them for 
//                    standalone hardware testing.) KS.
//     27th Aug 2006. Replaced use of nearbyint so the code will compile on
//                    Solaris. KS.
//     16th Sep 2006. Added BCD conversion routines. KS.
//     19th Feb 2007. Added Tokenize(). KS.
//     10th May 2007. Tokenize() is now more flexible, with additional
//                    arguments supporting quoted strings and end-of-line
//                    comments. KS.
//      4th Jun 2007. Added C++ string version of ExpandFileName(). KS.
//     17th Jan 2011. Include string.h and stdio.h. Needed from GCC 4.4.2. TJF.
//     16th Feb 2015. Include stdlib.h. Now needed for getenv() at atof(). KS.
//      5th Jun 2024. Replaced use of sprintf() by snprintf(). KS.
//      3rd Oct 2024. Split off Unix-specific routines to allow the rest
//                    (mostly string-processing routines) to be used under
//                    other systems like Windows. KS.
//
//    Copyright (c)  Anglo-Australian Telescope Board, 2005-2024.
//    Permission granted for use for non-commercial purposes.

static const char *rcsId="@(#) $Id: ACMM:HectorConfigUtility/TcsUtil.cpp,v 1.2+ 20-Nov-2020 13:47:31+11 ks $";
static void *use_rcsId = (0 ? (void *)(&use_rcsId) : (void *) &rcsId);

#include "TcsUtil.h"

#include <errno.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

using std::string;
using std::vector;

#ifdef DEBUG
#undef DEBUG
#endif
#define DEBUG printf

// -----------------------------------------------------------------------------

//                         F o r m a t  A r c s e c
/*!
 *   FormatArcsec() returns a string containing a formatted version of the
 *   double precision arcseconds value passed as an argument. 
 *
 *   \param  Arcsec  The value in arcseconds to be formatted.
 *
 *   \return A string containing the formatted arcsecond value.
 *
 *   \author Keith Shortridge, AAO.
 */
 
string TcsUtil::FormatArcsec (
   double Value)
{
   char Number[64];                     // Used to format numeric value
   
   double Seconds = Value;
   char Sign = ' ';
   if (Seconds < 0.0) {
      Seconds = -Seconds;
      Sign = '-';
   }
   int Hours = int(Seconds / 3600.0);
   Seconds -= double(Hours * 3600);
   int Minutes = int(Seconds / 60.0);
   Seconds -= double(Minutes * 60);
   int IntSeconds = int(Seconds);
   Seconds -= double(IntSeconds);
   int IntFrac = int((100.0 * Seconds) + 0.5);
   snprintf (Number,sizeof(Number),"%c%03d:%02d:%02d.%02d",Sign,Hours,Minutes,
                                                    IntSeconds,IntFrac);
   string FormattedString = Number;
   return FormattedString;
}

// -----------------------------------------------------------------------------

//                         F o r m a t  U i n t
/*!
 *   FormatUint() returns a string containing a formatted version of the
 *   unsigned int passed as an argument. 
 *
 *   \param  Value  The unsigned integer to be formatted.
 *
 *   \return A string containing the formatted integer value.
 *
 *   \author Keith Shortridge, AAO.
 */
 
string TcsUtil::FormatUint (
   unsigned int Value)
{
   char Number[64];                     // Used to format numeric value
   
   snprintf (Number,sizeof(Number),"%d",Value);
   string FormattedString = Number;
   return FormattedString;
}

// -----------------------------------------------------------------------------

//                         F o r m a t  I n t
/*!
 *   FormatUint() returns a string containing a formatted version of the
 *   int passed as an argument. 
 *
 *   \param  Value  The signed integer to be formatted.
 *
 *   \return A string containing the formatted integer value.
 *
 *   \author Keith Shortridge, AAO.
 */
 
string TcsUtil::FormatInt (
   int Value)
{
   char Number[64];                     // Used to format numeric value
   
   snprintf (Number,sizeof(Number),"%d",Value);
   string FormattedString = Number;
   return FormattedString;
}

// -----------------------------------------------------------------------------

//                         F o r m a t  U l o n g l o n g
/*!
 *   FormatUlonglong() returns a string containing a formatted version of the
 *   unsigned long long integer passed as an argument. Because long long
 *   integers are usually very long, this uses a comma formatting scheme.
 *
 *   \param  Value  The unsigned long long integer to be formatted.
 *
 *   \return A string containing the formatted integer value.
 *
 *   \author Keith Shortridge, AAO.
 */
 
string TcsUtil::FormatUlonglong (
   unsigned long long Value)
{
   char Number[64];                     // Used to format numeric value
   char Reverse[64];                    // Workspace for the comma formatting
   
   snprintf (Number,sizeof(Number),"%llu",Value);
   unsigned int Chars = strlen(Number);
   unsigned int Ind = 0;
   for (unsigned int I = 0; I < Chars; I++) {
      if ((I > 0) && ((I % 3) == 0)) {
         Reverse[Ind] = ',';
         Ind++;
      }
      Reverse[Ind] = Number[Chars - I - 1];
      Ind++;
   }
   for (unsigned int I = 0; I < Ind; I++) {
      Number[I] = Reverse[Ind - I - 1];
   }
   Number[Ind] = '\0';      
   string FormattedString = Number;
   return FormattedString;
}

//  ----------------------------------------------------------------------------

//                      E x p a n d  F i l e  N a m e
//
/*!  This routine is passed a file name that may have environment variables
 *   embedded in it.  If so, any such environment variable should be
 *   preceded by a '$'. For example, "$DCT_DIR/default.job". This routine
 *   replaces any such variables by their translated value. It also
 *   does a final pass through the resulting string to convert any double
 *   '/' characters into a single '/'. This means that DCT_DIR could be
 *   defined as, for example, either "/instsoft/iris2" or "/instsoft/iris2/"
 *   and in both cases "$DCT_DIR/default.job" will be expanded to
 *   "/instsoft/iris2/default.job". The environment variable names are assumed
 *   to be terminated a '/', a '.' or by the end of the string.
 *
 *   \param  Name         A C++ string giving the file name to be expanded.
 *   \param  ExpandedName The C++ string that will receive the expanded file
 *                        name.
 *
 *   \return              True if everything went OK, false if there was
 *                        a problem.
 *
 *   Note: This routine is a reworking of the original routine of the same
 *   name that handled C-style character arrays instead of C++ strings. It
 *   works the same way, except that it works recursively and will translate
 *   cases where an environment variable has a translation that includes
 *   an environment variable with a leading $ sign. The older version follows.
 */
 
bool TcsUtil::ExpandFileName (  //  True if OK, false if an error occured.
   const string& Name,          //  File name to be expanded.
   string& ExpandedName)        //  String to receive expanded name.
{
   bool Result = true;                          // Return code.
   string ResultString;                         // Local expanded string.
   string::size_type NotFound = string::npos;   // Named to make usage clearer.
   string::size_type DollarPosn;                // Position of dollar sign.
   string::size_type SlashPosn;                 // Position of '//' sequence.
   
   ResultString = Name;
   while ((DollarPosn = ResultString.find('$')) != NotFound) {
      string::size_type DelimPosn = ResultString.find_first_of("./",DollarPosn);
      string EnvVarName;
      if (DelimPosn == NotFound) {
         EnvVarName = ResultString.substr(DollarPosn + 1);
      } else {
         EnvVarName = ResultString.substr(DollarPosn + 1,
                                             DelimPosn - DollarPosn - 1);
      }
      char* NamePtr = getenv(EnvVarName.c_str());
            
      //  If we got a translation for the environment variable,
      //  we copy it into the expanded string.
            
      if (NamePtr == (char*) NULL) {
         Result = false;
         break;
      }
      ResultString.replace(DollarPosn,EnvVarName.size() + 1,NamePtr);
   }
      
   //  Now a pass through looking for '//', which we change to '/'
   
   while ((SlashPosn = ResultString.find("//")) != NotFound) {
      ResultString.replace(SlashPosn,2,"/");
   }
   
   //  Return the results
   
   ExpandedName = ResultString;
   return Result;
}         

//  ----------------------------------------------------------------------------

//                      E x p a n d  F i l e  N a m e
//
/*!  This routine is passed a file name that may have environment variables
 *   embedded in it.  If so, any such environment variable should be
 *   preceded by a '$'. For example, "$DCT_DIR/default.job". This routine
 *   replaces any such variables by their translated value. It also
 *   does a final pass through the resulting string to convert any double
 *   '/' characters into a single '/'. This means that DCT_DIR could be
 *   defined as, for example, either "/instsoft/iris2" or "/instsoft/iris2/"
 *   and in both cases "$DCT_DIR/default.job" will be expanded to
 *   "/instsoft/iris2/default.job". The environment variable names are assumed
 *   to be terminated a '/', a '.' or by the end of the string.
 *
 *   \param  Name         The address of a C-type nul-terminated string giving
 *                        the file name to be expanded.
 *   \param  ExpandedName The address of a C char array that will receive the
 *                        expanded file name, also nul-terminated.
 *   \param  Nchars       The size of the array in bytes whose address is
 *                        passed as ExpandedName.
 *
 *   \return              True if everything went OK, false if there was
 *                        a problem.
 *
 *   Note that this routine, which was taken from a utility routine used
 *   by the AAO's detector control task, works on C-type char arrays,
 *   rather than C++ strings. A new version is now available that has been
 *   reworked for C++ strings, and both are now included in TcsUtil.
 */
bool TcsUtil::ExpandFileName (  //  True if OK, false if an error occured.
   const char* Name,            //  Address of name to be expanded
   char* ExpandedName,          //  Address of string to receive expanded name
   int Nchars)                  //  Number of chars available at ExpandedName.
{
   int ExpIndex = 0;
   const char* NamePtr = Name;
   bool InEnvVar = false;
   bool Truncated = false;
   bool Invalid = false;
   int StartIndex = 0;
   
   for (;;) {
   
      //  Make sure there's room for at least one more character in the
      //  expanded string.
      
      if (ExpIndex >= Nchars) {
         Truncated = true;
         break;
      }
      char TheChar = *NamePtr++;
      if (TheChar == '$') {
      
         //  This is the start of an environment variable. We will copy
         //  the variable name into the expanded name string, but we note
         //  where it will start.
         
         if (InEnvVar) break;
         ExpandedName[ExpIndex++] = TheChar;
         StartIndex = ExpIndex;
         InEnvVar = true;
         
      } else if ((TheChar == '/') || (TheChar == '\0') || (TheChar == '.')) {
      
         //  Any of these may indicate the end of an environment variable
         
         if (InEnvVar) {
         
            InEnvVar = false;
         
            //  We've reached the end of an environment variable. We terminate
            //  the string in the expanded name so that we have something to
            //  pass to getenv(). (The need to terminate it is why we've
            //  copied it from Name - which is 'const', so we can't slip a
            //  temporary terminator into it there.)
            
            ExpandedName[ExpIndex++] = '\0';
            char* TranslatedName = getenv(&(ExpandedName[StartIndex]));
            
            //  If we got a translation for the environment variable,
            //  we copy it into the expanded string, starting at where
            //  we started the environment variable name. (Remember we
            //  left in the '$' in the expanded string in case we
            //  couldn't do the translation.
            
            if (TranslatedName == (char*) NULL) {
               Invalid = true;
               break;
            }
            ExpIndex = StartIndex - 1;
            for (;;) {
               if (ExpIndex >= Nchars) {
                  Truncated = true;
                  break;
               }
               if (*TranslatedName == '\0') break;
               ExpandedName[ExpIndex++] = *TranslatedName++;
            }
            if (Truncated) break;
         }
         
         //  Copy the delimiter into the final string, and break out
         //  if we've reached the end of the name string.
         
         ExpandedName[ExpIndex++] = TheChar;
         if (TheChar == '\0') break;
         
      } else {
      
         //  An ordinary character, so copy it over
         
         ExpandedName[ExpIndex++] = TheChar;
      }
   }

   //  Now a pass through looking for '//', which we change to '/'
   
   ExpIndex = 0;
   for (;;) {
      char TheChar = ExpandedName[ExpIndex];
      if (TheChar == '\0') break;
      if (TheChar == '/') {
         if (ExpandedName[ExpIndex + 1] == '/') {
            
            //  We've found a '//' case, so shuffle the rest of the
            //  string down one place and carry on.
            
            int Index = ExpIndex;
            do {
               ExpandedName[Index] = ExpandedName[Index + 1];
            } while (ExpandedName[++Index]);
         }
      }
      ExpIndex++;
   }
   
   //  If we have an untranslatable environment variable, it will be 
   //  at the end of the string we return. If the string was truncated,
   //  we make this clearer by appending an '...' at the end of the
   //  returned string.
   
   if (Truncated) {
      if (Nchars > 3) strcpy (&(ExpandedName[Nchars - 4]),"...");
   }
   
   return (!Truncated && !Invalid);
}         

// -----------------------------------------------------------------------------

//                       M a t c h  C a s e  B l i n d
/*!
 *   MatchCaseBlind() performs a case-insensitive comparision of two strings
 *   and returns true if they match.
 *
 *   \param  String1  The first of the two strings.
 *   \param  String2  The second of the two strings.
 *
 *   \return True if the strings match exactly in all but case.
 *
 *   \author Keith Shortridge, AAO.
 */
 
bool TcsUtil::MatchCaseBlind (  //  True if the strings match, false otherwise.
   const string& String1,       //  The first of the two strings.
   const string& String2)       //  The second of the two strings.
{
   bool Match = false;
   
   //  First, see if the strings are exactly the same. This should be quick.
   
   if (String1 == String2) {
      Match = true;
   } else {

      //  This is not particularly efficient, but it works.
   
      string UpperString1 = String1;
      string UpperString2 = String2;
      transform (UpperString1.begin(),UpperString1.end(),UpperString1.begin(),
                                                                     toupper);
      transform (UpperString2.begin(),UpperString2.end(),UpperString2.begin(),
                                                                     toupper);
      Match = (UpperString1 == UpperString2);
   }
   return Match;
}

// -----------------------------------------------------------------------------

//                       C o n v e r t  T o  B C D
/*!
 *   ConvertToBCD() converts an integer into a BCD value. This is suitable
 *   for converting the encoded Focus values used by TCS.
 *
 *   \param  Value  The integer to be converted.
 *
 *   \return The value converted into BCD format.
 *
 *   \author Keith Shortridge, AAO.
 */
 
unsigned int TcsUtil::ConvertToBCD (    //  The converted value.
   unsigned int Value)                  //  The value to be converted.
{
   //  I think this code could be more efficient, but it's fairly clear.
   //  It strips off each digit and adds it into the result. Each time
   //  through the loop, each digit is worth 16 times as much, hence
   //  the shifting by Bits.
   
   unsigned int Result = 0;
   unsigned int Bits = 0;
   while (Value > 0) {
      Result += (Value % 10) << Bits;
      Bits += 4;
      Value /= 10;
   }
   return Result;
}

// -----------------------------------------------------------------------------

//                       C o n v e r t  F r o m  B C D
/*!
 *   ConvertFromBCD() converts a BCd value into an integer. This is suitable
 *   for converting the encoded Focus values used by TCS.
 *
 *   \param  Value  The BCD value to be converted.
 *
 *   \return The BCD value converted into an integer.
 *
 *   \author Keith Shortridge, AAO.
 */
 
unsigned int TcsUtil::ConvertFromBCD (    //  The converted value.
   unsigned int Value)                    //  The BCD value to be converted.
{
   //  I think this code could be more efficient, but it's fairly clear.
   //  It strips off each digit and adds it into the result. Each time
   //  through the loop, each digit is worth 10 times as much.
   
   unsigned int Result = 0;
   unsigned int Factor = 1;
   while (Value > 0) {
      Result += (Value % 16) * Factor;
      Factor *= 10;
      Value >>= 4;
   }
   return Result;
}


// -----------------------------------------------------------------------------

//                          T o k e n i z e
/*!
 *   Tokenize() takes a string and splits it up into tokens, separated by 
 *   one of a set of delimiters. By default the delimiter is a blank. This
 *   routine returns a vector of strings. (Actual strings, not pointers,
 *   so destroying the vector destroys the strings as well.) This supports
 *   quoted strings, and the use of a specified comment character that
 *   introduces an end of line comment that the tokenizer ignores.
 *
 *   \param  InputString  A string containing a set of tokens separated by
 *                        delimiters.
 *   \param  Tokens       A vector of strings, to which this routine adds
 *                        the tokens found in InputString. Note that they are
 *                        added, so Tokens will probably need a clear() call
 *                        if reused.
 *   \param  Delimiters   A string containing one or more delimiter characters.
 *                        This argument is optional, and if omitted the routine
 *                        assumes tokens in InputString are delimited by 
 *                        blanks.
 *   \param  Quotes       A string containing one or more characters used
 *                        for quoted strings. If omitted, the default is
 *                        a single quote and a double quote. If a token starts
 *                        with a quote character, it continues until the
 *                        matching quote character is found.
 *   \param  Comments     A string containing one or more characters used
 *                        to introduce end end of line comment. Any token
 *                        that starts with one of these characters, 
 *                        together with the rest of the line, is ignored.
 *   \param  Blanks       A string containing characters that serve a blank
 *                        space. This can be omitted, in which case a single
 *                        blank is assumed. If the argument is supplied, it
 *                        can be used to include tab characters a blanks,
 *                        for example. 
 *
 *   \author Keith Shortridge, AAO (but see below).
 *
 *   The code found here was originally taken from code by Alavoor Vasudevan, 
 *   found  in the C++ Programming How-to at 
 *   
 *   www.oopweb.com/CPP/Documents/CPPHOWTO/Volume/C++Programming-HOWTO
 *
 *   I have since messed it about unrecognisably, to support quoted
 *   strings, comments, and to make it handle perverse cases such as
 *   '1, 2' and '1,, 2' where the use of blank as a delimiter and
 *   also conceptual blank space got very confusing. I believe that
 *   although the code now looks much messier than the original rather
 *   elegant few lines, it gets closer to doing what I found I wanted.
 */

void TcsUtil::Tokenize(
   const string& InputString,
   vector<string>& Tokens,
   const string& Delimiters,
   const string& Quotes,
   const string& Comments,
   const string& Blanks)
{
   string::size_type Pos;                     // Start position for each token.
   string::size_type LastPos = 0;             // Final token delimiter position.
   string::size_type Length;                  // Length of the string.
   string::size_type NotFound = string::npos; // To make the use obvious.

   Length = InputString.size();
   
   // Skip delimiters at beginning.
   
   Pos = InputString.find_first_not_of(Blanks);
   while (Pos != NotFound  && LastPos != NotFound)
   {
      //  Look at the start character.  If it's a comment character, stop now.
      
      char Char = InputString.at(Pos);
      if (Comments.find(Char) != NotFound) break;
      
      //  See if it's a quote character.
      
      if (Quotes.find(Char) != NotFound) {
      
         //  It is, so any token runs from one past to just before any
         //  matching quote, if any. If no match was found, we use all
         //  of the string.
         
         Pos++;
         LastPos = InputString.find(Char,Pos);
         if (LastPos == NotFound) {
            Tokens.push_back(InputString.substr(Pos));
         } else {
            Tokens.push_back(InputString.substr(Pos, LastPos - Pos));
            LastPos++;
         }
         
      } else {
      
         //  Not a quote character, so the token goes from where we are 
         //  up to the next delimiter.
         
         LastPos = InputString.find_first_of(Delimiters,Pos);
         Tokens.push_back(InputString.substr(Pos, LastPos - Pos));
      }
      
      //  At this point, we want to skip any blanks and up to one
      //  non-blank delimiter, starting at LastPos. This is messy, but
      //  without it the ' ,' in '2 ,3' gets counted, or else the
      //  null token in '1,,3' gets skipped.
      
      int NonBlankDelims = 0;
      while (LastPos < Length) {
         Char = InputString.at(LastPos);
         if (Blanks.find(Char) == NotFound) {
            if (Delimiters.find(Char) == NotFound) break;
            if (NonBlankDelims++ > 0) break;
         }
         LastPos++;
      }
      Pos = LastPos;
      if (Pos >= Length) Pos = NotFound;
   }
}

#if defined(__unix__) || defined(__APPLE__)

#include <unistd.h>
#include <sched.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <sys/mman.h>
#include <sys/utsname.h>
#include <strings.h>

// -----------------------------------------------------------------------------

//                         G e t  E r r n o  T e x t
/*!
 *   GetErrnoText() returns a string containing a description of the
 *   current errno value. The idea is that it can be used to help generate
 *   a message describing just what happened when a system routine returns
 *   bad status.
 *
 *   \return A string describing the current errno value.
 *
 *   \author Keith Shortridge, AAO.
 */
 
string TcsUtil::GetErrnoText (void)
{   
   string ErrnoString = strerror(errno);
   return ErrnoString;
}

// -----------------------------------------------------------------------------

//                         T r a n s i e n t  E r r o r
/*!
 *   Many system calls can return prematurely, usually because they have
 *   been interrupted by a signal. In some cases, they can just be
 *   called again. TransientError() checks the current errno value to see
 *   if is a value such as EAGAIN or EINTR which indicates that a system
 *   routine can be retried. So this should be used when a system routine
 *   returns a -1 value. If TransientError() returns true, retry the system
 *   call.
 *
 *   \return True if the erroring system call can be retried.
 *
 *   \author Keith Shortridge, AAO.
 */
 
bool TcsUtil::TransientError (void)
{
   return ((errno == EAGAIN) || (errno == EINTR));
}

// -----------------------------------------------------------------------------

//                     R e a d  S o c k e t  D a t a
/*!
 *   Reading from a socket requires slightly more than just issuing a
 *   read() call for the given file descriptor. Socket reads can be
 *   interrupted and need to be retried, and if a socket is set non-
 *   blocking the fact that a select() call shows that data is available
 *   does not prove that all the data is ready - a read may not
 *   get all the data in this case. This routine handles these various
 *   considerations. It will read the required amount of data from the
 *   specified socket, even if the socket is non-blocking. Note that
 *   this means that this routine effectively blocks until all the data
 *   is available, so this should not be used if a number of sockets
 *   are being monitored unless you are confident that delaying to read
 *   all the data from one socket as soon as it shows as readable is
 *   not going to be a problem. (If it is, you need a much more complex
 *   scheme.) 
 *
 *   \param  SocketFd    Gives the file descriptor open on the socket.
 *   \param  Buffer      The address of the buffer into which data is to
 *                       be read.
 *   \param  Bytes       The number of bytes to be read. Do not just pass
 *                       the size of the buffer, unless the buffer really is
 *                       to be filled, as this routine keeps reading until
 *                       it has read all of the specified byte count.
 *   \param  TimeoutMsec A timeout value in milliseconds. If the specified
 *                       number of bytes have not been read in this time,
 *                       this routine returns an error value of -1. If a 
 *                       timeout of zero is specified, the routine waits
 *                       indefinitely.
 *   \param  ErrorText   A string left unchanged unless there is an error. If
 *                       an error occurs, ErrorText is set to a description
 *                       of the error.
 *
 *   \return The number of bytes read, if all went well (which will always
 *           be equal to the Bytes argument). If an error occurs, -1 is
 *           returned.
 *
 *   \author Keith Shortridge, AAO.
 */

int TcsUtil::ReadSocketData (           // Returns bytes read, or -1
   int SocketFd,                        // Socket file descriptor
   char* Buffer,                        // Data buffer
   unsigned int Bytes,                  // Number of bytes to read
   unsigned int TimeoutMsec,            // Timeout in msec. (0 => indefinite)
   std::string& ErrorText)              // Unchanged except on error
{
   unsigned int BytesToRead = Bytes;
   char* BufferPtr = Buffer;
   int ReturnValue = Bytes;

   //  This code assumes that the socket is set non-blocking, in which
   //  case we may need to repeat the read until we get all the data we
   //  want. (Which means the timeout applies to each read separately.)
   //  If the socket is blocking, the effect will be that we wait until
   //  some data is available, but then assume that we can read all the 
   //  specified data.
    
   while (BytesToRead > 0) {
   
      if (TimeoutMsec > 0) {
      
         //  A timeout was specified, so use select() to wait for data to
         //  be ready on the socket. This allows a timeout to be supplied.
         
         fd_set TestDescriptors;               // Used to test file descriptors
         struct timeval Timeout;
         bool DataReady = false;
         FD_ZERO(&TestDescriptors);
         FD_SET(SocketFd,&TestDescriptors);
         Timeout.tv_sec = TimeoutMsec / 1000;
         Timeout.tv_usec = (TimeoutMsec % 1000) * 1000;
         
         //  This loop, potentially repeating the select call, allows for the
         //  possibility that select() is interrupted. We come out of it
         //  either on a non-transient error or when data is ready.
         
         while (!DataReady) {
            int Status = select(SocketFd + 1,
                                 &TestDescriptors,NULL,NULL,&Timeout);
            if (Status == 0) {
               ErrorText = "Timeout error waiting to read from socket fd "
                                                      + FormatInt(SocketFd);
               break;
            } else if (Status < 0) {
               if (!TransientError()) {
                  ErrorText = "Select error waiting to read from socket fd "
                                                      + FormatInt(SocketFd);
                  break;
               }
            } else {
               DataReady = true;
            }
         }
         
         //  If data wasn't ready, then we've had an error or timeout.
         
         if (!DataReady) {
            ReturnValue = -1;
            break;
         }
      }
      
      //  Now read as much of the data as we have available.
      
      int BytesRead = read (SocketFd,BufferPtr,BytesToRead);
      if (BytesRead == 0) {
         ErrorText = "End of file on socket fd " + FormatInt(SocketFd);
         ReturnValue = -1;
         break;
      } else if (BytesRead < 0) {
         if (TransientError()) {
            BytesRead = 0;
         } else {
            ErrorText = "Error reading " + FormatUint(BytesToRead) +
                      " bytes from socket fd " + FormatInt(SocketFd);
            ReturnValue = -1;
            break;
         }
      }
      BytesToRead -= BytesRead;
      BufferPtr += BytesRead;
   }
   return ReturnValue;  
}

// -----------------------------------------------------------------------------

//                     W r i t e  S o c k e t  D a t a
/*!
 *   Writing to a socket needs to allow for the possibility that
 *   a write() call will be interrupted and need to be retried, and
 *   even for the possibility that a single write() call may not
 *   transfer all the data. This routine handles these considerations,
 *   writing the specified data to the socket and only returning when
 *   all the data has been written. (Note that this is not the same as
 *   a guarantee that it has been received, only that it has been
 *   written to the socket.)
 *
 *   \param  SocketFd    Gives the file descriptor open on the socket.
 *   \param  Buffer      The address of the buffer from which data is to
 *                       be written.
 *   \param  Bytes       The number of bytes to be written.
 *   \param  ErrorText   A string left unchanged unless there is an error. If
 *                       an error occurs, ErrorText is set to a description
 *                       of the error.
 *
 *   \return The number of bytes written, if all went well (which will always
 *           be equal to the Bytes argument). If an error occurs, -1 is
 *           returned.
 *
 *   \author Keith Shortridge, AAO.
 */
 
int TcsUtil::WriteSocketData (          // Returns bytes written, or -1
   int SocketFd,                        // Socket file descriptor
   const char* Buffer,                  // Data buffer
   unsigned int Bytes,                  // Number of bytes to write
   std::string& ErrorText)              // Unchanged except on error
{
   unsigned int BytesToWrite = Bytes;
   const char* BufferPtr = Buffer;
   int ReturnValue = Bytes;
   
   while (BytesToWrite > 0) {
      int BytesWritten = write(SocketFd,BufferPtr,BytesToWrite);
      if (BytesWritten < 0) {
         if (TransientError()) {
            BytesWritten = 0;
         } else {
            ReturnValue = -1;
            ErrorText = "Error writing " + FormatUint(BytesToWrite) +
                        " bytes to socket fd " + FormatInt(SocketFd);
            break;
         }
      }
      BufferPtr += BytesWritten;
      BytesToWrite -= BytesWritten;
   }
   return ReturnValue;
}

// -----------------------------------------------------------------------------

//                         R e a d  T S C
/*!
 *   All modern Intel PC architecture systems support a time stamp counter,
 *   the TSC, which is incremented by one for each tick of the system clock.
 *   So for a 1Ghz system, the TSC goes up by one each nanosecond. This
 *   can provide a very accurate way of measuring short intervals. This
 *   routine returns the current value of the TSC. On systems that do not
 *   have a TSC, it returns a plausible imitation. 
 *
 *   \return An unsigned long long value giving the current TSC.
 *
 *   \author Keith Shortridge, AAO.
 */
 
unsigned long long TcsUtil::ReadTSC (void)

//  There are two versions of this routine. One is for PC architecture
//  machines - such as the TCS target system - which have a genuine time
//  stamp counter, and one for all the other systems. For the PC systems,
//  we assume the __i386__ compiler flag will be set (note that you can tell
//  which system-specific flags are set, at least for gcc, by using
//  cpp -dM), and we generate the necessary assembler instruction to read
//  the TCS. For other systems, which will be running in simulation and
//  where extreme accuracy will not really be an issue, we fall back on
//  using gettimeofday().

#ifdef __i386__
{
   //  This code snippet comes from the realfeel() code as distributed by
   //  Andrew Morton. Realfeel was originally written by Mark Hahn - I don't
   //  know which of them wrote this code, but I'm grateful to them both.
   
    unsigned long long tsc;
    __asm__ __volatile__("rdtsc" : "=A" (tsc));
    return tsc;
}
#else
{
   //  For non-PC systems, we use gettimeofday(), which provides a value in
   //  seconds and microseconds. We turn this into something approximating
   //  a TSC value for a 1GHz system. Note: it's important to have all the
   //  calculations done in 64 bit, using long long, which is why Sec and
   //  USec are used as intermediate long long variables. Note that
   //  in 2005 even (CurrentTime.tv_sec * 1000000) will overflow the
   //  32 bit limit.
   
   struct timeval CurrentTime;
   gettimeofday (&CurrentTime,NULL);
   unsigned long long Nanosec;
   unsigned long long Microsec;
   unsigned long long Sec = CurrentTime.tv_sec;
   unsigned long long USec = CurrentTime.tv_usec;
   Microsec = (Sec * 1000000) + USec;
   Nanosec = Microsec * 1000;
   return Nanosec;
}
#endif

// -----------------------------------------------------------------------------

//          S e t  A s  R e a l  T i m e  H i g h  P r i o r i t y
/*!
 *   SetAsRealTimeHighPriority() allows a suitably priveleged task to
 *   set itself up as a high priority real-time task with memory locked
 *   into real memory. This is needed for tasks that need the best
 *   real-time response that the operating system can provide. On Linux
 *   this requires that the task be running with root privileges.
 *   If the task fails to set itself up properly (usually because it
 *   does not have the necessary privilege, or because the operating
 *   system does not support the facility, at least as coded here),
 *   an error message is returned in the ErrorText string and the
 *   routine returns false. If all goes OK, it returns true.
 *
 *   \param  ErrorText   A string left unchanged unless there is an error. If
 *                       an error occurs, ErrorText is set to a description
 *                       of the error.
 *
 *   \return True if all went well, false if an error occurred.
 *
 *   \author Keith Shortridge, AAO.
 */

bool TcsUtil::SetAsRealTimeHighPriority(// Returns true if OK, false otherwise
   std::string& ErrorText)              // Unchanged except on error

#ifdef __linux__
{
   //  Under Linux, a suitably privileged task can set itself to use
   //  'fifo' scheduling at the highest priority level. It can also
   //  lock all its memory both now and in the future.
   
   bool ReturnStatus = false;
   struct sched_param SchedParam;
   int Invoke;
   SchedParam.sched_priority = sched_get_priority_max(SCHED_FIFO);
   Invoke = sched_setscheduler (0,SCHED_FIFO,&SchedParam);
   if (Invoke) {
      ErrorText = "Error setting real-time scheduling: " + GetErrnoText();
   } else {
      Invoke = mlockall (MCL_CURRENT | MCL_FUTURE);
      if (Invoke) {
         ErrorText = "Error locking memory: " + GetErrnoText();
      } else {
         ReturnStatus = true;
      }
   }
   return ReturnStatus;
}

#else

{
   //  Under non-Linux operating systems, what we want to do may well
   //  be possible, but it isn't required for TCS test purposes.
   
   ErrorText = "Cannot set up as a real-time task under this OS";
   return false;
}
   
#endif

//  ----------------------------------------------------------------------------

//                  S c h e d u l e  Q u a n t u m
//
/*!  ScheduleQuantum() returns what is really no more than a best guess at
 *   the time quantum used by the operating system for scheduling
 *   purposes. The delay we request for the FAST loop is rounded down with
 *   this quantum taken into account. The value returned is in microseconds.
 *
 *   \return An estimate of the operating system scheduling quantum in
 *           microseconds.
 */
 
unsigned int TcsUtil::ScheduleQuantum (void)
{
   //  We do this in a somewhat ad hoc way. I don't know of a proper way
   //  to get this value. We simply assume that Linux 2.6 kernels and
   //  above have a 1 msec quantum and earlier releases use 10 msec.
   //  Other systems don't really matter for TCS purposes, so we use
   //  10 msec for them.
   
   unsigned int QuantumUsec = 10000;
   struct utsname UnameBuffer;
   if (uname(&UnameBuffer) >= 0) {
      if (!strcmp(UnameBuffer.sysname,"Linux")) {
      
         //  This is Linux. Now look at the release. We're only interested
         //  in the first two parts of the number, which can be something
         //  like "2.4.21-40.ELsmp" (from aaolxp).
         
         char Release[10];
         unsigned int I;
         int DotCount = 0;
         strncpy (Release,UnameBuffer.release,sizeof(Release));
         Release[sizeof(Release) - 1] = '\0';
         for (I = 0; I < sizeof(Release); I++) {
            if (Release[I] == '.') {
               DotCount++;
               if (DotCount == 2) {
                  Release[I] = '\0';
               }
            }
         }
         if (atof(Release) >= 2.6) {
            QuantumUsec = 1000;
         }
      }
   }
   return QuantumUsec;
}

#endif

// -----------------------------------------------------------------------------

//                          T e s t  P r o g r a m
//
//  This brief test program is only generated if the pre-processor variable
//  TcsUtil_TEST is defined.  It is hardly an exhaustive test, and it
//  leaves the socket I/O routines completely untested.

#ifdef TcsUtil_TEST

#include <iostream>
#include <istream>

using std::cout;
using std::cin;

int main ()
{
   unsigned int Uvalue = 42;
   int Value = -42;
   cout << Uvalue << " formats as " << TcsUtil::FormatUint(Uvalue) << '\n';
   cout << Value << " formats as " << TcsUtil::FormatInt(Value) << '\n';
   unsigned long long Llvalue = 1;
   for (int I = 0; I < 20; I++) {
      cout << Llvalue << " formats as " << 
                              TcsUtil::FormatUlonglong(Llvalue) << '\n';
      Llvalue *= 10;
   }
   unsigned long long TSC,TSC1;
   TSC = TcsUtil::ReadTSC();
   sleep(1);
   TSC1 = TcsUtil::ReadTSC();
   cout << "TSC was " << TcsUtil::FormatUlonglong(TSC) << '\n';
   cout << "1 sec later, TSC was " << TcsUtil::FormatUlonglong(TSC1) << '\n';
   cout << "A change of " << TcsUtil::FormatUlonglong(TSC1 - TSC) << '\n';
   if (TcsUtil::MatchCaseBlind("abcd","AbCd")) {
      cout << "Correct: abcd matches AbCd\n";
   } else {
      cout << "*** Error comparing abcd with AbCd\n";
   }
   string FileName = "$HOME/filename.dat";
   char ExpandedName[64];
   string ExpandedString;
   if (!TcsUtil::ExpandFileName(FileName,ExpandedString)) {
      cout << "Error expanding " << FileName << '\n';
   } else {
      cout << FileName << " expands to '" << ExpandedString << "'\n";
   } 
   if (!TcsUtil::ExpandFileName(FileName.c_str(),
                                   ExpandedName,sizeof(ExpandedName))) {
      cout << "Error expanding " << FileName << '\n';
   } else {
      cout << FileName << " expands to '" << ExpandedName << "'\n";
   } 
   double Arcsec = (123.0 * 3600.0) + (45.0 * 60.0) + 12.34;
   cout << " 123:45:12.34 formats as " << TcsUtil::FormatArcsec(Arcsec) << '\n';
   cout << "-123:45:12.34 formats as " << 
                                         TcsUtil::FormatArcsec(-Arcsec) << '\n';
   if (TcsUtil::MatchCaseBlind("abcd","AbCd")) {
      cout << "Correct: abcd matches AbCd\n";
   } else {
      cout << "*** Error comparing abcd with AbCd\n";
   }
   if (!TcsUtil::MatchCaseBlind("abce","AbCd")) {
      cout << "Correct: abce does not match AbCd\n";
   } else {
      cout << "*** Error comparing abce with AbCd\n";
   }
   string ErrorText;
   if (TcsUtil::SetAsRealTimeHighPriority(ErrorText)) {
      cout << "Successfully set up as a real-time task\n";
   } else {
      cout << "Failed to set up as a real-time task\n";
      cout << ErrorText << '\n';
   }
   cout << "Scheduler time quantum estimated as " 
            << TcsUtil::ScheduleQuantum() << " usec\n";
   cout << "1234 formats in BCD as " << std::hex
                                       << TcsUtil::ConvertToBCD(1234) << "\n";
   cout << "9876 formats in BCD as " << std::hex 
                                       << TcsUtil::ConvertToBCD(9876) << "\n";
   cout << "0 formats in BCD as " << std::hex << 
                                       TcsUtil::ConvertToBCD(0) << "\n";
   cout << "5 formats in BCD as " << std::hex << 
                                       TcsUtil::ConvertToBCD(5) << "\n";
   bool Failed = false;
   for (unsigned int Value = 0; Value < 10000; Value++) {
      if (TcsUtil::ConvertFromBCD(TcsUtil::ConvertToBCD(Value)) != Value) {
         cout << "Error in conversion of " << Value << " to BCD and back\n";
         Failed = true;
         break;
      }
   }
   if (!Failed) {
      cout << "Conversions to BCD and back verified for range 0..9999\n";
   }
   vector<string> Tokens;
   
   TcsUtil::Tokenize("  ",Tokens);
   TcsUtil::Tokenize("Ha,Dec, Dome",Tokens,", ");
   TcsUtil::Tokenize("Windscreen,Focus",Tokens,",");
   vector<string>::iterator Iter;
   for(Iter = Tokens.begin(); Iter != Tokens.end(); Iter++) {
      cout << "'" << *Iter << "' ";
   }
   cout << "\n";
   if (Tokens.size() != 5) cout << "Error in tokenizer. " << Tokens.size() 
                                                      << " items found\n";
   else cout << "Correct number of tokens found\n";
   Tokens.clear();
   TcsUtil::Tokenize("  ' This is a quoted string'",Tokens,", ","'\"");
   for(Iter = Tokens.begin(); Iter != Tokens.end(); Iter++) {
      cout << "'" << *Iter << "' ";
   }
   cout << "\n";
   if (Tokens.size() != 1) cout << "Error in tokenizer. " << Tokens.size() 
                                                      << " items found\n";
   else cout << "Correct number of tokens found\n";
   
   //  This section is an interactive test of the tokenizer. It is
   //  usually disabled for automatic tests.
   
   if (false) {
      string Line;
      for (;;) {
         cout << "String: \n";
         if (!getline(cin,Line)) break;
         Tokens.clear();
         TcsUtil::Tokenize(Line,Tokens,", ","'\"","#"," ");
         cout << "Total of " << Tokens.size() << " tokens found\n";
         for(Iter = Tokens.begin(); Iter != Tokens.end(); Iter++) {
            cout << "'" << *Iter << "' ";
         }
         cout << "\n";
      }
   }
   
   
}

#endif   
   
