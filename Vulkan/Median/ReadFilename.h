//
//                    R e a d  F i l e n a m e . h
//
//  Function:
//    ReadFilename() prompts for a filename and returns one entered by the user
//
//  Description:
//    ReadFilename() provides a way for a command line program - ie one run
//    from a terminal - to prompt the user for the name of a file. What makes
//    this more than just a "write a prompt, read a line" routine is that it
//    supports filename completion. If the user enters a TAB character, the
//    routine attempts to complete what the user has entered so far with the
//    name of any existing file that matches, to the extent that this can be
//    determined unambiguously. If the user enters a control-D character,
//    the program lists all the files it can find that match what has been
//    entered so far. The result is a reasonably close match to that provided
//    by the command lines of most shells, such as tcsh.
//
//  Call:
//    int ReadFilename (Prompt, Line, Len)
//
//  Where:
//    Prompt is a C-style character string, nul-terminated, that is output
//           as a prompt to the user. Normally, it would be something along
//           the lines of "Enter filename: ".
//    Line   is the address of the start of an array of char, into which the
//           resulting file specification, entered by the user, is returned.
//           It will always be terminated by a final nul character.
//    Len    is the number of bytes in the Line array - it will normally
//           be just sizeof(Line).
//
//  Returns:
//    The routine returns zero if all goes well, -1 if there is a problem.
//    Usually, the only problems occur when trying to set the terminal into
//    the non-echo mode required to allow the file completion code work.
//
//  Note:
//    o The routine assumes that stdin is a terminal.
//    o The routine does not support editing using cursor keys.
//    o Filenames that include spaces can be entered normally, with no need
//      to escape the spaces - the routine assumes only one filename is being
//      entered, so does not treat space as a delimiter.
//    o Because it confuses things significantly, the wildcard character '*'
//      is simply ignored - this routine won't accept it. If you want a
//      filename specification that includes wildcards, just prompt for it as
//      an ordinary string.
//    o There's no attempt to check that the file exists, or that it has a
//      correctly formatted file name. You may want the name of file that
//      you intend to create - you may even intend to create the whole of
//      a specified directory structure.
//
//  Author: Keith SHortridge, K&V (Keith@Knave&Varlet.com.au)
//
//  History:
//     22nd Jan 2021. Original version.

#ifndef __ReadFilename__
#define __ReadFilename__

#include <stdlib.h>

int ReadFilename (const char* Prompt, char* Line, size_t Len);

#endif
