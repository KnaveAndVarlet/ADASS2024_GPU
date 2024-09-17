//
//                     T c s  U t i l . h
//
//  Function:
//     A class providing a number of utility routines for the TCS.
//
//  Description:
//     The TcsUtil class provides a grab-bag of assorted utility
//     routines that have been found to be needed by more than one 
//     TCS modules.
//
//  Author(s): 
//     Keith Shortridge, AAO.
// 
//  History:
//     18th Nov 2005.  Original version. KS.
//      5th Dec 2005.  Added ExpandFileName(). KS.
//     12th Dec 2005.  Added FormatArcsec(). KS.
//      4th May 2006.  Added MatchCaseBlind(). KS.
//     17th May 2006.  Added SetAsRealTimeHighPriority(). KS.
//     22nd May 2006.  Added EncodeReason(). KS.
//     25th May 2006.  Oops. EncodeReason() should have been static. Fixed. KS.
//     29th May 2006.  Added ScheduleQuantum(). KS.
//     14th Aug 2006.  Moved EncodeReason() to the TCS_DRAMATask code, allowing
//                     this collection of general utilities to be independent
//                     of DRAMA. (Which makes it easier to use them for 
//                     standalone hardware testing.) KS.
//     18th Aug 2006.  Included a dummy EncodeReason() for the moment so older
//                     code doesn't break. KS.
//     16th Sep 2006.  Added BCD conversion routines. KS.
//     19th Feb 2007.  Added Tokenize(). KS.
//     10th May 2007.  Tokenize() is now more flexible, with additional
//                     arguments supporting quoted strings and end-of-line
//                     comments. KS.
//      4th Jun 2007.  Added C++ string version of ExpandFileName(). KS.
//
//  RCS id:
//     "@(#) $Id: ACMM:HectorConfigUtility/TcsUtil.h,v 1.2+ 20-Nov-2020 13:47:31+11 ks $"
//
//    Copyright (c)  Anglo-Australian Telescope Board, 2005,2006,2007.
//    Permission granted for use for non-commercial purposes.

#ifndef __TcsUtil__

#define __TcsUtil__

#include <string>
#include <vector>

//!  TcsUtil is a class containing an unrelated collection of utility routines.

/*!  The TcsUtil class is a repository for a collection of unrelated routines
  *  that have been found to be generally useful for different TCS modules.
  *  It's really a subroutine library cast as a class. All the methods are
  *  static, so it's a class that you aren't expected to create an
  *  instance of, although that hasn't been prevented.
  *
  *  \author  Keith Shortridge, AAO
  */
  
class TcsUtil {
public:
   //!  Format the latest errno value into a string.
   static std::string GetErrnoText (void);
   //!  Format an unsigned integer into a string.
   static std::string FormatUint (unsigned int Value);
   //!  Format a signed integer into a string.
   static std::string FormatInt (int Value);
   //!  Format an unsigned long long integer into a string.
   static std::string FormatUlonglong (unsigned long long Value);
   //!  Format a value in arcseconds into a string.
   static std::string FormatArcsec (double Value);
   //!  Return true if the last errno value represents a transient error.
   static bool TransientError (void);
   //!  Read from a socket, allowing for transient errors etc.
   static int ReadSocketData (int SocketFd,char* Buffer,unsigned int Bytes,
                             unsigned int TimeoutMsec, std::string& ErrorText);
   //!  Write to a socket, allowing for transient errors.
   static int WriteSocketData (int SocketFd,const char* Buffer,
                                   unsigned int Bytes, std::string& ErrorText);
   //!  Return the current Time Stamp Counter (or a suitable emulation).  
   static unsigned long long ReadTSC (void);
   //!  Expand a file name that contains environment variable names.      
   static bool ExpandFileName (const std::string& Name, 
                                             std::string& ExpandedName);
   //!  Older version of ExpandFileName that uses C char arrays.      
   static bool ExpandFileName (const char* Name, 
                                 char* ExpandedName, int Nchars);
   //!  Perform a case-insensitive comparision of two strings.
   static bool MatchCaseBlind (const std::string& String1, 
                                     const std::string& String2);
   //!  Set the current task as a high-priority real-time task.
   static bool SetAsRealTimeHighPriority (std::string& ErrorText);
   //!  Get the operating system minimum scheduling delay in microsecs.
   static unsigned int ScheduleQuantum (void);
   //!  Convert an integer into BCD.
   static unsigned int ConvertToBCD (unsigned int Value);
   //!  Convert from BCD to an integer.
   static unsigned int ConvertFromBCD (unsigned int Value);
   //!  Split a string into separate tokens.
   static void Tokenize (const std::string& Str,
       std::vector<std::string>& Tokens,const std::string& Delimiters = " ",
       const std::string& Quotes = "\"\'", const std::string& Comments = "",
       const std::string& Blanks = " \t");
   //!  Temporary placeholder for a now deprecated routine.
   static std::string EncodeReason (int /* Reason */) { return "unknown"; }
};      

#endif      
