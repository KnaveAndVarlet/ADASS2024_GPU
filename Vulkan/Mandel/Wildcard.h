//
//                        W i l d c a r d   M a t c h
//
//                 W i l d c a r d  M a t c h  C a s e  B l i n d
//
//  Function: A few routines that do wildcard matches on character strings.
//
//  Description:
//     WildcardMatch() compares a test string against a pattern that can
//     contain wildcard characters. At present, the only wildcard accepted
//     is '*', which can stand for any number of characters. (That number
//     can be zero.) There can be any number of such characters in the 
//     pattern string. This is a pretty standard simple wildcard routine,
//     but it serves for most purposes I need. WildcardMatchCaseBlind()
//     does the same thing, but its character comparisions ignore case. So,
//     for example, WildcardMatch("*abc*","xyzabcdef") is a match, as is
//     WildcardMatch("*abc*","abc"), WildcardMatch("*","abc") or
//     WildcardMatchCaseBlind("*ABC*","xyzabc"). WildcardMatch("","") (two
//     null strings) is true, as is WildcardMatch("*","").

#ifndef __WildcardMatch__
#define __WildcardMatch__

extern "C" {

bool WildcardMatch (const char* Pattern, const char* TestString);
bool WildcardMatchCaseBlind (const char* Pattern, const char* TestString);

}

#endif
