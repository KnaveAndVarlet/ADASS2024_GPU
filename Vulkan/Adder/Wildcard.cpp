//
//                        W i l d c a r d   M a t c h
//
//                 W i l d c a r d  M a t c h  C a s e  B l i n d
//
//  Function: A few routines that do wildcard matches on character strings.
//
//  Description:
//     WildcardMatch() compares a test string against a pattern that can
//     contain wildcard characters. At present, the only wildcards accepted
//     are '*', which can stand for zero or more characters, and '?' which
//     stands for any single character. There can be any number of such 
//     characters in the pattern string. This is a pretty standard simple
//     wildcard routine, but it serves for most purposes I need. 
//     WildcardMatchCaseBlind() does the same thing, but its character 
//     comparisions ignore case. So, for example, 
//     WildcardMatch("*abc*","xyzabcdef") is a match, as is
//     WildcardMatch("*abc*","abc"), WildcardMatch("*","abc") or
//     WildcardMatchCaseBlind("*ABC*","xyzabc"). WildcardMatch("","") (two
//     null strings) is true, as is WildcardMatch("*","").
//
//  Call:
//     bool WildcardMatch (const char* Pattern, const char* TestString)
//     bool WildcardMatchCaseBlind (const char* Pattern,const char* TestString)
//
//     where Pattern and TestString are the addresses of nul-terminated
//     C strings. Pattern may contain any number of '*' or '?' wild card 
//     characters. Both routines return true to indicate a match, false 
//     otherwise.
//
//  Algorithm:
//     The code looks at the pattern and extracts the first string of
//     non-wild characters it finds. In the pattern "*abc*" this is just
//     the "abc" in the middle. It then looks to see if it can find this
//     string in the test string. If it does so, it splits the two strings
//     into three parts. There is the central part, which is the matching
//     non-wild string. There is the left hand part, which are the characters
//     preceding this central part in both strings, and the right hand part,
//     which are the characters following the central part in both strings.
//     It then calls itself recursively to compare the left and right 
//     parts of the two strings. If all three parts match, then there is a
//     match and that's it. If not, it looks to see if there are other matches
//     to the central part further down the test string, and carries on until
//     it no longer finds a match. In which case, that's it too. 
//      
//     Eventually, each chain of comparisions will end up with a zero 
//     length portion of either the pattern or the test string, or with a
//     pattern that consists only of wildcard characters. These are the 
//     'bail-out' conditions for the recursion. If either portion is
//     zero length whether we have a match
//     depends on whether both or only one is zero, and whether the remaining
//     portion of the pattern is completely wild or not.
//
//  Coding notes:
//     I only wrote this because - to my surprise - I couldn't find any
//     nicely packaged code that did this simple task. The standard regexec()
//     and related routines use POSIX regular expressions, and didn't quite
//     do what I wanted. The Boost library is said to provide this, but as is
//     its way, it does so in a very complicated way and in any case I didn't
//     want to bring in Boost just for a simple problem like this. In the end,
//     I did it myself. I chose to use standard C strings instead of C++
//     std::strings because this can then be used for both C and C++ strings
//     (string::c_str() is just sitting there). I didn't want to have to
//     allocate copies of strings (eg to make upper case versions of both
//     test and pattern before doing the comparison). It would have been
//     nice to pass nul-terminated strings to the various internal routines,
//     but that would require modifying them in place or making copies, so
//     strings are all passed around using a start address and a length.
//
//  Author: K. Shortridge, AAO.
//
//  History:
//     21st May 2013. Original version, KS.
//      4th Jun 2013. Added support for the '?' character. KS.
//     21st Jan 2021. Fixed a pedantic compiler warning about potential
//                    uninitialised variables (not an actual problem, but
//                    very hard for a compiler to realise that). Some minor
//                    corrections to comments. KS.
//
//  Copyright (c) Australian Astronomical Observatory (AAO), (2013).
//  Permission is hereby granted, free of charge, to any person obtaining a 
//  copy of this software and associated documentation files (the "Software"), 
//  to deal in the Software without restriction, including without limitation 
//  the rights to use, copy, modify, merge, publish, distribute, sublicense, 
//  and/or sell copies of the Software, and to permit persons to whom the 
//  Software is furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in 
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE SOFTWARE.

#include "Wildcard.h"
  
#include <stdio.h>
#include <string.h>
#include <ctype.h>

//  ----------------------------------------------------------------------------

//  FindNonWildString() searches the first Length characters from String up,
//  looking for the first sequence of non-wild characters. If it finds such 
//  a sequence, it returns true and returns the start offset of the sequence
//  in FirstFound and the offset of the last character in the sequence
//  in LastFound. (Both are offsets from the start of the string, ie from
//  String.) So FindNonWildString("**ABC**DEF**",12,&First,&Last) would
//  find the sequence "ABC" and would set First to 2 and Last to 4.

static bool FindNonWildString (const char* String, int Length, 
                                   int* FirstFound, int* LastFound)
{
   bool Found = false;
   
   //  Look for the first non-wild character.
   
   int Offset = 0;
   while (Offset < Length) {
      if (String[Offset] != '*' && String[Offset] != '?') {
         *FirstFound = Offset;
         break;
      }
      Offset++;
   }
   
   //  If we didn't find one, Offset will be equal to Length.
   
   if (Offset < Length) {
   
      //  We did find one, look through the rest of the characters for
      //  the end of the sequence - the next wild character.
      
      Found = true;
      while (Offset < Length) {
         if (String[Offset] == '*' || String[Offset] == '?') {
            *LastFound = Offset - 1;
            break;
         }
         Offset++;
      }
      
      //  If there wasn't one, the end of the sequence must be the end of 
      //  the string itself.
      
      if (Offset >= Length) *LastFound = Length - 1;
   }
   return Found;
}

//  ----------------------------------------------------------------------------

//  WildMatch() is passed a pattern that consists entirely of wild cards
//  characteers ('*' or '?'). It returns true if this string would match
//  a test string that contains TLength characters. The pattern starts at
//  the address given by Pattern and is PLength long. The test comes down
//  simply to the number of '?' characters in the pattern: the test string
//  must have at least one character for each '?' that appears in the pattern.
//  If the number of characters in the test string is more than the number of
//  '?' characters in the pattern, the pattern must have at least one '*'
//  for there to be a match. Note that if PLength is greater than the number
//  of '?' in the pattern, it must contain at least one '*'.

static bool WildMatch (const char* Pattern, int PLength, int TLength)
{
   bool Match = false;
   int MinChars = 0;
   for (int I = 0; I < PLength; I++) if (Pattern[I] == '?') MinChars++;
   if (TLength == MinChars) Match = true;
   else if (PLength > MinChars) Match = true;
   return Match;
}

//  ----------------------------------------------------------------------------

//  FindSubstringInString() looks for a given substring (in all cases, this
//  is going to be the central non-wild sequence of characters in the Pattern
//  found by FindNonWildString()) of length SubLength starting at Substring, 
//  in the Length characters starting at String (which will be a subset of the
//  TestString). The comparison can be case-blind or not. If the substring is
//  found, this returns true and sets StartOffset to the offset from the start
//  of String at which the substring was found.
//
//  Note that this is the only routine in this code that actually compares
//  characters, so is the only place where CaseBlind makes any difference,

static bool FindSubstringInString (const char* Substring, int SubLength,
             const char* String, int Length, bool CaseBlind, int* StartOffset)
{
   bool Found = false;
   if (SubLength > 0) {
      int Offset = 0;
      
      //  Look at each possible start position in the string - which depends
      //  on the length of the substring.
        
      int LastOffset = Length - SubLength;
      while (Offset <= LastOffset) {
         bool Match = true;
         
         //  Probably faster to have separate loops for the two 'case' cases..
         
         if (CaseBlind) {
            for (int Index = 0; Index < SubLength; Index++) {
               if (toupper(Substring[Index]) != 
                                   toupper(String[Offset + Index])) {
                  Match = false;
                  break;
               }
            }
         } else {
            for (int Index = 0; Index < SubLength; Index++) {
               if (Substring[Index] != String[Offset + Index]) {
                  Match = false;
                  break;
               }
            }
         }
         if (Match) {
            Found = true;
            *StartOffset = Offset;
            break;
         }
         Offset++;
      }
   }
   return Found;
}

//  ----------------------------------------------------------------------------

//  WildcardMatchInternal() does the real work of the algorithm, as described
//  in the introductory comments. This is the portion that gets called
//  recursively, It compares that portion of the pattern that starts from
//  Pattern and is PLength characters long with that portion of the test
//  string that starts at TestString and is TLength characters long, the
//  case sensitivity of any comparisions being controlled by CaseBlind.
      
static bool WildcardMatchInternal (const char* Pattern, int PLength, 
                     const char* TestString, int TLength, bool CaseBlind)
{
   bool Match = false;
   bool Done = false;
   
   //  Find the 'central' part of the pattern - a substring that only
   //  contains non-wild characters - if there is one. (We need to know
   //  this now, because if we bail out now, the return code depends on 
   //  whether the remaining pattern is completely wild or not.)
   
   int First = 0,Last = 0;
   bool IsWild = !FindNonWildString(Pattern,PLength,&First,&Last);
   
   //  Check the bail-out conditions - either of the two strings is zero
   //  length, or the pattern string contains nothing but wild characters,
   //  and set the final return code accordingly. The cases where the
   //  pattern is purely wild are handled by WildMatch().
   
   if (TLength == 0) {
      Done = true;
      if (PLength == 0) {
         Match = true;
      } else if (IsWild) {
         Match = WildMatch(Pattern,PLength,TLength);
      }
   } else {
      if (PLength == 0) {
         Done = true;
      } else if (IsWild) {
         Match = WildMatch(Pattern,PLength,TLength);
         Done = true;
      }
   }
   if (!Done) {
   
      //  Not finished, so we have a central non-wild part of the pattern
      //  that might also be found in the test string. We look for it. We
      //  might find it more than once, of course (think pattern "*bb" and
      //  test string "abbcbbdbb" where the only match is when the very last
      //  'bb' in the test string lines up against the 'bb' of the pattern).
      //  We work down the test string, each time looking at the TestLength
      //  characters from TestString[Index] to the end of the test string,
      //  starting at the start of the string (Index = 0), of course. The
      //  Non-wild central part of the pattern is Length long, so we can stop
      //  when there are fewer characters than that left in the test string.
      
      int Length = Last - First + 1;
      int Index = 0;
      int TestLength = TLength;
      while (Length <= TestLength) {
         int StartOffset;
         
         //  The central string (with no wild characters) is Length long
         //  and starts at Pattern[First]. Look for the next match in what's
         //  left of the test string, which is from TestString[Index] and is
         //  now TestLength long. If not found, no match at all, quit.
          
         bool Found = FindSubstringInString (&Pattern[First],
                 Length,&TestString[Index],TestLength,CaseBlind,&StartOffset);
         if (!Found) break;
         
         //  There is a match. We want the actual offset into TestString, not
         //  the one from Index, so correct for that.
         
         StartOffset += Index;
         
         //  We now know center part of the test string (starts at 
         //  TestString[StartOffset], and is Length long) matches the non-wild
         //  part of the pattern that starts at Pattern[First] and is also 
         //  Length long).  We have a complete match if the left hand part
         //  of the test string matches the left hand part of the pattern, 
         //  and if the same is true of the right hand sides. Call ourselves 
         //  recursively to test the left and right hand parts. Getting the 
         //  lengths for the right hand side correct is the hardest part of
         //  the code here.
         
         bool LeftMatches = WildcardMatchInternal(Pattern,First,
                                           TestString,StartOffset,CaseBlind);
         if (LeftMatches) {
            bool RightMatches = WildcardMatchInternal(
                        &Pattern[First+Length],PLength - First - Length,
                          &TestString[StartOffset + Length],
                                 TLength - StartOffset - Length,CaseBlind);
            if (RightMatches) {
               Match = true;
               break;
            }
         }
         
         //  If both left and right matched, we'd have broken out of the loop.
         //  They don't, so we see if there's another match later on in the
         //  pattern to the non-wild part of the pattern. If there is, it
         //  must be after TestString[StartOffset]. So the remaining part of
         //  the test string is from TestString[StartOffset+1]. It's length
         //  is from there to the end of the test string, of course.
           
         Index = StartOffset + 1;
         TestLength = TLength - Index;
      }
   }
   return Match;
}

//  ----------------------------------------------------------------------------

//                       W i l d c a r d  M a t c h
//
//  The main WildcardMatch routine. Sets up the initial call to 
//  WildcardMatchInternal, with CaseBlind set false, and that's it.

bool WildcardMatch (const char* Pattern, const char* TestString)
{
   return WildcardMatchInternal (Pattern,strlen(Pattern),TestString,
                                                     strlen(TestString),false);
}

//  ----------------------------------------------------------------------------

//              W i l d c a r d  M a t c h  C a s e  B l i n d
//
//  The main WildcardMatchCaseBlind routine. Sets up the initial call to 
//  WildcardMatchInternal, with CaseBlind set true, and that's it.
      
bool WildcardMatchCaseBlind (const char* Pattern, const char* TestString)
{
   return WildcardMatchInternal (Pattern,strlen(Pattern),TestString,
                                                     strlen(TestString),true);
}

//  ----------------------------------------------------------------------------

//                              M a i n
//
//  A simple test program that exercises some of the facilities of 
//  WildcardMatch() and WildcardMatchCaseBlind().

#ifdef WILDCARD_MATCH_TEST

bool WildcardMatchTest (const char* Pattern, const char* TestString)
{
   printf ("'%s' vs '%s' ",Pattern,TestString);
   return WildcardMatch (Pattern,TestString);
}
      
bool WildcardMatchTestCaseBlind (const char* Pattern, 
                                                    const char* TestString)
{
   printf ("'%s' vs '%s' ",Pattern,TestString);
   return WildcardMatchCaseBlind (Pattern,TestString);
}
      
int main()
{
   int Errors = 0;
   if (WildcardMatchTest("**bb","bbbxbbxxbb")) printf ("Matches OK\n");
   else printf ("OOps! Error #%d\n",++Errors);
   if (WildcardMatchTest("**bb*c","bbbc")) printf ("Matches OK\n");
   else printf ("OOps! Error #%d\n",++Errors);
   if (WildcardMatchTestCaseBlind("**Bb*C","bbbc")) printf ("Matches OK\n");
   else printf ("OOps! Error #%d\n",++Errors);
   if (WildcardMatchTest("AB","AB")) printf ("Matches OK\n");
   else printf ("OOps! Error #%d\n",++Errors);
   if (WildcardMatchTestCaseBlind("AB","ab")) printf ("Matches OK\n");
   else printf ("OOps! Error #%d\n",++Errors);
   if (WildcardMatchTest("","")) printf ("Matches OK\n");
   else printf ("OOps! Error #%d\n",++Errors);
   if (WildcardMatchTest("**??**","ab")) printf ("Matches OK\n");
   else printf ("OOps! Error #%d\n",++Errors);
   if (WildcardMatchTest("*","")) printf ("Matches OK\n");
   else printf ("OOps! Error #%d\n",++Errors);
   if (WildcardMatchTest("*?","abc")) printf ("Matches OK\n");
   else printf ("OOps! Error #%d\n",++Errors);
   if (WildcardMatchTest("abc???","abcdef")) printf ("Matches OK\n");
   else printf ("OOps! Error #%d\n",++Errors);
   if (WildcardMatchTest("?","")) printf ("OOps! Error #%d\n",++Errors);
   else printf ("No match - OK\n");
   if (WildcardMatchTest("?","ab")) printf ("OOps! Error #%d\n",++Errors);
   else printf ("No match - OK\n");
   if (WildcardMatchTest("**ab*c","bbbc")) printf("OOps! Error #%d\n",++Errors);
   else printf ("No match - OK\n");
   if (WildcardMatchTest("","bbbc")) printf ("OOps! Error #%d\n",++Errors);
   else printf ("No match - OK\n");
   if (WildcardMatchTest("**ab*c","")) printf ("OOps! Error #%d\n",++Errors);
   else printf ("No match - OK\n");
   if (WildcardMatchTest("a","b")) printf ("OOps! Error #%d\n",++Errors);
   else printf ("No match - OK\n");
   if (WildcardMatchTest("A","a")) printf ("OOps! Error #%d\n",++Errors);
   else printf ("No match - OK\n");
   
   if (Errors == 0) printf ("\nNo errors\n\n");
   else printf ("%d Error(s) ***\n",Errors);
   
   return 0;
}

#endif     
   
