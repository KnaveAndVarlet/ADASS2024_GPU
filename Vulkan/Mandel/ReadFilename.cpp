//
//                    R e a d  F i l e n a m e . c p p
//
//  Function:
//    ReadFilename() prompts for a filename and returns one entered by the user
//
//  Description:
//    ReadFilename() provides a way for a command line program - ie one run
//    from a terminal - to prompt the user for the name of a file. What makes
//    this more than just a "write a prompt, read a line" routine is that it
//    supports filename completion. For full details, see the comments at the
//    start of ReadFilename.h
//
//
//  Author: Keith SHortridge, K&V (Keith@Knave&Varlet.com.au)
//
//  History:
//     22nd Jan 2021. Original version.

#include "ReadFilename.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <glob.h>

#include <sys/time.h>
#include <termios.h>

// -----------------------------------------------------------------------------
//
//                     I n t e r n a l  R o u t i n e s

static void CheckTilde (char* Line,int MaxIdx, int* LineSt,int* PathSt);
static void HandleCtrlD (const char* Prompt,char* Line,size_t Len,int Idx);
static void HandleTab (char* Line,size_t Len,int* Idx);


// -----------------------------------------------------------------------------
//
//                         R e a d  F i l e n a m e
//
//  This is ReadFilename() itself. It outputs the prompt that it
//  is passed, then sets the terminal into non-canonical/no echo mode so it
//  has control voer what the user sees, as opposed to just echoing characters
//  as they're typed, and starts to read characters from the user. It handles
//  TAB and ^D characters to provide filename completion, and finally
//  returns the entered filename. If all goes well, it returns zero, but
//  can return -1 to indicate an error changing the terminal mode.
//
//    Prompt is a C-style character string, nul-terminated, that is output
//           as a prompt to the user. Normally, it would be something along
//           the lines of "Enter filename: ".
//    Line   is the address of the start of an array of char, into which the
//           resulting file specification, entered by the user, is returned.
//           It will always be terminated by a final nul character.
//    Len    is the number of bytes in the Line array - it will normally
//           be just sizeof(Line).

int ReadFilename (const char* Prompt, char* Line, size_t Len)
{
   int Code = 0;
   char Del = 127;
   char Backup = 0x8;
   int CtrlD = 0x4;
   struct termios OldMode;
   struct termios NewMode;

   //  Put out the prompt.
   
   printf ("%s",Prompt);
   
   //  Get the current terminal attributes, save them, and apply a new
   //  set of attributes based on the current ones, but with canonical mode
   //  (reading a line at a time) disabled, automatic echoing disabled,
   //  and the timer disabled and input set to one byte at a time.
   
   Code = tcgetattr (0, &OldMode);
   if (Code == 0) {
      memcpy (&NewMode, &OldMode, sizeof(struct termios));
      NewMode.c_lflag &= ~(ICANON | ECHO);
      NewMode.c_cc[VTIME] = 0;
      NewMode.c_cc[VMIN] = 1;
      Code = tcsetattr (0, TCSANOW, &NewMode);
      if (Code == 0) {
         int Idx = 0;
         int MaxIdx = Len - 1;
         
         //  Now just read characters one by one until we get a newline (or
         //  EOF). Most characters are just put directly into Line and
         //  echoed, but some are handled explicitly.
         
         for (;;) {
            char Char = fgetc(stdin);
            if (Char == '\n' || Char == EOF) {
               fputc (Char,stdout);
               Line[Idx] = '\0';
               break;
            }
            if (Char == Del) {
            
               //  Delete is handled by backing up over the last character,
               //  blanking it by writing a space, and then backing up over
               //  that space.
               
               if (Idx > 0) {
                  fputc (Backup,stdout);
                  fputc (' ',stdout);
                  fputc (Backup,stdout);
                  Idx--;
               }
            } else if (Char == '\t') {
            
               //  A TAB triggers filename completion, and is complex enough
               //  to need its own routine.
               
               HandleTab (Line,Len,&Idx);
               
            } else if (Char == CtrlD) {
            
               //  ^D lists all files that match the spec entered so far,
               //  and also needs its own routine.
               
               HandleCtrlD (Prompt,Line,Len,Idx);
               
            }  else if (Char == '*') {
            
               //  We simply ignore any attempt to enter a wildcard character.
               //  This confuses things too much.
               
            } else {
            
               //  Normal characters go into Line and are echoed, so long
               //  as there is room left in the buffer.
               
               if (Idx < MaxIdx) {
                  Line[Idx++] = Char;
                  fputc (Char,stdout);
               }
            }
         }
      }
      
      //  Restore the original terminal settings.
      
      int RestoreCode = tcsetattr (0, TCSANOW, &OldMode);
      if (Code == 0) Code = RestoreCode;
   }
   return Code;
}

// -----------------------------------------------------------------------------
//
//                          C h e c k  T i l d e
//
//  CheckTilde() is simply some common code needed by HandleTab() and
//  HandleCtrlD(), both of which have to handle the way glob() deals with
//  file specifications beginning with '~'. If glob() is passed the GLOB_TILDE
//  flag - as it is by this routine - it recognises the '~user/' as the home
//  directory of the named user, and '~/' as the home directory of the
//  current user. However, it take it upon itslelf to expand these into the
//  absolute file path when it returns the list of matching files, which is
//  useful, but not what's really wanted here. This routine assumes the Line
//  it is passed starts with one of these '~' specifications, and returns
//  the index in Line just after that first delimiting '/'. and also returns
//  the index of the character in the paths returned by glob() just after
//  the expansion of that '~' string. That's messy, but an example may help:
//
//  If my user name is 'fred' and my home directory is '/Users/fred/', then
//  if I pass '~fred/myfile*' to glob, glob will return results like:
//  '/Users/fred/myfile.o'
//  '/Users/fred/myfile.cpp'
//  etc.
//  Rather than have the user suddenly see the '~fred' that was typed
//  suddenly turn into '/Users/fred'. we'd like to be able to restore the
//  original '~' form. To do that, we need to know how many characters of
//  the paths returned by glob() need to be replaced by how many characters
//  in the original spec. In this case, the first 6 characters ('~fred/') of
//  the original spec need to replace the first 12 characters ('/Users/fred/')
//  of the paths returned by glob().
//
//  It's those two counts that CheckTilde() works out. In this case, it
//  will set LineSt to 6 and PathSt to 12. It's then up to the calling
//  routine to make use of that information.
//

void CheckTilde (char* Line,int MaxIdx,int* LineSt,int* PathSt)
{
   *LineSt = 0;
   *PathSt = 0;
   int Ich = 0;
   
   //  Find the first '/' that delimits the '~' spec at the start of Line.
   
   while (Line[Ich] != '\0' && Line[Ich] != '/') Ich++;
   if (Ich + 1 < MaxIdx) {
   
      //  Now, pass just that part of the Line spec to glob(). glob()
      //  will return just the expansion of the '~' spec, just as it
      //  will do when matching wildcrd specifications. And the
      //  length of that returned expansion will be what we want.
      
      char SaveCh = Line[Ich + 1];
      Line[Ich + 1] = '\0';
      glob_t HomeGlob;
      if (glob(Line,GLOB_TILDE,NULL,&HomeGlob) == 0) {
         if (HomeGlob.gl_pathc == 1) {
            *LineSt = Ich + 1;
            *PathSt = strlen(HomeGlob.gl_pathv[0]);
         }
      }
      Line[Ich + 1] = SaveCh;
      globfree(&HomeGlob);
   }
}

// -----------------------------------------------------------------------------
//
//                       H a n d l e  C t r l  D
//
//  HandleCtrlD() is called when a ctrl-D is entered. This returns all
//  the existing files that match the specification entered so far
//  into Line. Idx should be passed as the index of the next free
//  character in Line, so the characters entered so far are Line[0]
//  through Line[Idx - 1] inclusive. Len should be the size of the
//  char array that starts at Line. Prompt should be the prompt string
//  originally passed to ReadFilename(), which this routine will need if
//  it is to re-issue the prompt after listing the matching files.

void HandleCtrlD (const char* Prompt,char* Line,size_t Len,int Idx)
{
   int MaxIdx = Len - 1;
   if ((Idx + 1) < MaxIdx) {
   
      //  Get all the files that match the string so far by appending
      //  a wildcard '*' to the characters already in Line and passing
      //  that to glob().
      
      Line[Idx] = '*';
      Line[Idx + 1] = '\0';
      glob_t PGlob;
      if (glob (Line,GLOB_TILDE,NULL,&PGlob) == 0) {
      
         //  glob() returns the number of matching paths in .gl_pathc,
         //  and their values are in the char*[] array at gl_pathv.
         
         int GCount = PGlob.gl_pathc;
         if (GCount > 0) {
         
            //  Really, all we should need to do is print out the list
            //  of paths. However, we have to deal with the way glob()
            //  expands filenames starting with '~'. See comments to
            //  CheckTilde().
            
            int LineSt = 0;
            int PathSt = 0;
            if (Line[0] == '~' && PGlob.gl_pathv[0][0] != '~') {
               CheckTilde (Line,MaxIdx,&LineSt,&PathSt);
            }
            printf ("\n");
            
            //  Print each path as two parts. The original '~' spec at
            //  the start, then the rest of the path as returned by glob().
            //  For the - probably more usual - case where the path didn't
            //  start with '~', the first part is null and the second part
            //  will be the whole of the path returned by glob(), as
            //  LineSt and PathSt will both be zero.
            
            for (int IPath = 0; IPath < GCount; IPath++) {
               for (int I = 0; I < LineSt; I++) fputc(Line[I],stdout);
               int LPath = strlen(PGlob.gl_pathv[IPath]);
               for (int I = PathSt; I < LPath; I++) {
                  fputc(PGlob.gl_pathv[IPath][I],stdout);
               }
               printf ("\n");
            }
            
            //  Now re-issue the prompt, with the string the user has
            //  entered so far.
            
            printf ("%s",Prompt);
            for (int I = 0; I < Idx; I++) fputc(Line[I],stdout);
         }
      }
      globfree (&PGlob);
   }
}

// -----------------------------------------------------------------------------
//
//                           H a n d l e  T a b
//
//  HandleTab() is called when a TAB is entered. This looks at all the existing
//  files that match the string as entered so far in Line. Idx should be passed
//  as the index of the next free character in Line, so the characters entered
//  so far are Line[0] through Line[Idx - 1] inclusive. Len should be the size
//  of the char array that starts at Line. This routine looks at the list of
//  matching files, works out the starting characters they all have in common,
//  and appends these to the characters already in Line so as to complete the
//  entered file specification as far as is possible unambiguously. Usually,
//  the value of Idx will need to be updated as a result of this expansion,
//  which is whay Idx needs to be passed by address.

void HandleTab (char* Line,size_t Len,int* Idx)
{
   int MaxIdx = Len - 1;
   if ((*Idx + 1) < MaxIdx) {
   
      //  Get all the files that match the string so far by appending
      //  a wildcard '*' to the characters already in Line and passing
      //  that to glob(). This is pretty much the same as is done in
      //  HandleCtrlD - see comments in there.
      
      Line[*Idx] = '*';
      Line[*Idx + 1] = '\0';
      glob_t PGlob;
      if (glob (Line,GLOB_TILDE,NULL,&PGlob) == 0) {
         int GCount = PGlob.gl_pathc;
         if (GCount > 0) {
         
            //  Now see how long the common section of all these returned
            //  paths is. That's the expanded filespec that we need.
            //  We work through all the characters in the paths, starting
            //  at the first character (Ind = 0) and carrying on until we
            //  find a path where the characters at that position differ.
            //  (It's enough to test each path against the first.) When
            //  we break out, Ind is the index of the first non-matching
            //  character.
            
            int Ind = -1;
            bool Match = true;
            do {
               Ind++;
               char Ch = PGlob.gl_pathv[0][Ind];
               if (Ch == '\0') break;
               for (int IPath = 0; IPath < GCount; IPath++) {
                  if (PGlob.gl_pathv[IPath][Ind] != Ch) {
                     Match = false;
                     break;
                  }
               }
            } while (Match);
            
            //  An easy way to do this is just to copy all the characters
            //  from 0 to Ind-1 from that first path returned by glob()
            //  into Line, replacing what was there. But we have to allow
            //  for the way glob() handles specs starting with '~'. (See
            //  HandleTab() comments) Here, we leave any '~' spec at the
            //  start of Line untouched, and just copy over the path
            //  characters from where the expanded '~' spec ends to the
            //  last matching character.
            
            int LineSt = 0;
            int PathSt = 0;
            if (Line[0] == '~' && PGlob.gl_pathv[0][0] != '~') {
               CheckTilde (Line,MaxIdx,&LineSt,&PathSt);
            }
            for (int I = PathSt; I < Ind; I++) {
               if (LineSt < MaxIdx) Line[LineSt++] = PGlob.gl_pathv[0][I];
            }
            
            //  And now we finally echo those appended characters to the
            //  terminal, and update the index value in Idx.
            
            for (int I = *Idx; I < LineSt; I++) fputc (Line[I],stdout);
            *Idx = LineSt;
            if (*Idx > MaxIdx) *Idx = MaxIdx;
         }
         
      }
      globfree (&PGlob);
   }
}


// -----------------------------------------------------------------------------

//                             M a i n
//
//  A trivial test program, built if READ_FILE_TEST is defined.

#ifdef READ_FILE_TEST

int main()
{
   char line[1024];
   int size = sizeof(line);
   ReadFilename ("Enter filename: ",line,size);
   printf ("%s\n",line);
   return 0;
}

#endif

// -----------------------------------------------------------------------------

/*                       P r o g r a m m i n g  N o t e s

   o There is little reason for this to be in C++ rather than C, other than
     the convenience of C++ comments, being able to use bool, and being able
     to declare variables just before needing to use them. But that seems
     enough. Basically, though, the code is C. I even wondered if the .h
     file should use C-style comments and declare ReadFile() as extern 'C'.
     In the end, I didn't see the point.
 
   o Ignoring '*' characters is a bit of a cop-out. They could be handled
     using glob(), but it's complicated. Maybe in a later version? (The main
     issue is how to mke sure what the user sees is actually correct.)
 
   o This doesn't provide line editing using cursor keys, but if a cursor
     key is pressed the results aren't really what's expected. It would be
     nice to trap and ignore cursor keys, but since these generate key
     sequences rather than a single character, that's been put off for a
     later version as well.
*/
