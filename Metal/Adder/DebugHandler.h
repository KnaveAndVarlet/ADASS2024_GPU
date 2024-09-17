//
//                                 D e b u g  H a n d l e r . h
//
//   Introduction:
//
//      This implements a simple but relatively flexible way of controlling debug output from a
//      C++ program. This imagines that a program consists of a number of sub-systems, each given a
//      name, and each sub-system supports varying levels of debug output, each of which is also
//      given a name. The idea is that at any given time you may be interested in getting debug
//      output from one or more particular sections of the program, but not from all. And sometimes
//      you may not want debug output at all. The Debug handler code lets you associate a separate
//      DebugHandler object with each subsystem of your code, and provides ways to enable or
//      disable the various debug levels.
//
//   Overview:
//
//      Supply the sub-system name as a constructor argument, or using SetSubSystem().
//      Set the various level names that will be recognised using SetLevelNames().
//      Enable a set of levels using EnableLevels() or disable them using DisableLevels().
//      Log debug messages for a named level using Log() or Logf(). These are only logged
//      if the named level is active.
//
//   In much more detail:
//
//      The Debug Handler for a sub-system can be just a declared variable, or it can be created
//      using 'new' when the sub-system is initialised, eg:
//
//          DebugHandler TheDebugHandler("Test");
//      or
//          DebugHandler* DebugHandlerPtr = new DebugHandler("Test");
//
//      where both examples create a new DebugHandler and sets the sub-system name to 'Test'.
//
//      A DebugHandler needs to be told the set of level names to be used. These should be passed
//      to it using a call to SetLevelNames(), with the names passed as a comma-separated list, eg:
//
//          TheDebugHandler.SetLevelNames("Graphics,Timing,Data,Diagnostics");
//
//      A call to the SetLevelNames() method supplies a comma-separated list of the level names
//      used with this debug handler. (And a ListLevels() call will return that list.)
//      SetLevelNames() should normally only be called once. It resets the Debug Handler completely,
//      with all the named levels flagged as inactive.
//
//      A call to the EnableLevels() method passes a comma-separated set of strings, each of the
//      form "SubSystem.level". So, for example,
//
//          TheDebugHandler.EnableLevels("Test.Graphics,Test.Data");
//
//      would enable the Graphics and Data levels, but leave the Timing and Diagnostics levels
//      disabled.
//
//      As the program runs, calls can be made to the Log() method, supplying a string giving
//      a named level and a text string. If the level matches any of the active levels, the
//      DebugHandler outputs the text string to standard output. For example:
//
//          TheDebugHandler.Log("Setup","This is a debug message connected with setup.");
//
//      This message will be output if the 'Setup' level has been enabled. If you need a more
//      detailed message, the Logf() call provides printf() type formatting, eg:
//
//          TheDebugHandler.Log("Setup","Stage %d of setup took %.2f msec.",Stage,Msec);
//
//      The most complex part of all this is to do with the strings that can be passed to
//      EnableLevels() Both the sub-system and level part of the string can include wildcards
//      ('*' for any number of characters, or none, and '?' for a single character). If the
//      subsystem matches that supplied in the DebugHandler's constructor, the DebugHandler adds
//      any level that matches the level part of the string to its active list. If the subsystem
//      part of the string - the bit before the '.' - is missing, all DebugHandlers will respond
//      to the specification.
//
//      Moreover, a '!' prefixed to a 'subsystem.level' specification reverses the effect.
//      Specifications are parsed in sequence, so a later specification can override an
//      earlier one. For example,
//
//          TheDebugHandler.EnableLevels("Test.D*,!Test.Data");
//
//      would first enable any level in Test that started with 'D', eg both Data and Diagnostics,
//      but then would disable just Data. That example is pretty silly, but could be useful if
//      a system had a number of levels that started with 'D' and you want to enable all of them
//      except for Data.
//
//      Levels can be de-activated using a call to DisableLevels(), which is passed a string
//      identical to that for EnableLevels() but which removes any matching levels supported by the
//      DebugHandler from the active list.
//
//      Debug code could also use this to reduce excessive debug output. For example, after 20
//      times round the same loop, a program could explicitly switch of a named set of levels.
//      And it could re-enable them later.
//
//      A call to Active() returns true if a specified level is active, and can be used to
//      bypass a complete block of debug code if that is more efficient than relying on the
//      tests performed by each Log() call. It could even be used by code to enable other levels
//      explicitly if a specific named level is active. This is quite a flexible scheme!
//
//  Use with multiple sub-systems:
//
//      The idea is that the same string used to enable debugging levels can be passed to every
//      DebugHandler used in a program. (Usually, there would be some mechanism that allowed such a
//      string to be passed to each sub-system, which would then call the EnableLevels() method of
//      its own DebugHandler. So, for example, if a program had three sub-systems, a 'view', a
//      'model' and a 'controller' (to pick a possible combination), you might have:
//
//          DebugHandler ViewDebugHandler("View");                  (in the View code setup)
//          DebugHandler ControllerDebugHandler("Controller");      (in the Controller code setup)
//          DebugHandler ModelDebugHandler("Model");                (in the Model code setup)
//
//      and maybe:
//
//          ViewDebugHandler.SetLevelNames("Setup,Timing,Windows");
//          ControllerDebugHandler.SetLevelNames("Setup,Events");
//          ModelDebugHandler.SetLevelNames("Setup,Timing,Data,Diagnostics");
//
//      then passing the string "*.Setup" or just "Setup", to the EnableLevels() call of each
//      of the three handlers would enable Setup level debugging in all three systems.
//      Passing the string "View.Setup,Model.Timing" would enable Setup level debugging in
//      the View subsystem, and also Timing level debugging in the Model subsystem.
//
//      This works particularly well with command line programs (and this code was always intended
//      to be used in this way). A single string parameter set on the command line can be used
//      to set debug levels for all the separate parts of a program.
//
//      CheckLevels() takes a list of levels such as those passed to Enable/DisableLevels() to
//      check whether these will be recognised or not. It returns a list with all those that
//      are not recognised. In principle, this allows CheckLevels() to be called for a series
//      of different DebugHandlers, each being passed the string rejected by the previous handler.
//      If you end up with a non-blank string, there may be a problem. In practice, this may be
//      tricky to use for reasons of program structure.
//
//  Author(s): Keith Shortridge, K&V  (Keith@KnaveAndVarlet.com.au)
//
//  History:
//     16th Jan 2021.  Original version. KS.
//      9th Jun 2024.  Added SetSubSystem() and GetSubSystem(). KS.
//     21st Jun 2024.  Log() and Logf() now allow for a blank subsystem. KS.
//      1st Jul 2024.  Reformatted to 4-space indents, introductory commenting extended, some
//                     routines re-named for clarity. Added use of '!' in level specifications. KS.
//     14th Aug 2024.  Added CheckLevels(). Removed the programming note saying such a routine
//                     would be a good idea. KS.
//
//  Note:
//
//     This code needs to be compiled using at least -std=c++11, because it uses C++11 style
//     iteration through containers.
//
//     SetLevelNames() was originally called LevelsList(), which was a silly name. LevelsList()
//     is still supported - it now just calls SetLevelNames() - but should no longer be used.
//     Similarly, EnableLevels() and DisableLevels() were originally called SetLevels() and
//     UnsetLevels(), which was less silly but still was easy to confuse with SetLevelNames().
//     Again, the old names are still supported, but their use is discouraged.

#ifndef __DebugHandler__
#define __DebugHandler__

#include "Wildcard.h"
#include "TcsUtil.h"

#include <string>
#include <vector>

#include <stdio.h>
#include <stdarg.h>

class DebugHandler {
public:
    
    //  Constructor, takes optional sub-system name.
    
    DebugHandler (const std::string& SubSystem = "") {
        SetSubSystem(SubSystem);
    }
    
    //  Destructor has nothing to do - no resources to release.
    
    ~DebugHandler () {}
    
    //  SetSubSystem() sets the subsystem name - usually used because the constructor didn't.
    
    void SetSubSystem (const std::string& SubSystem) {
        I_SubSystem = SubSystem;
    }
    
    //  GetSubSystem() returns the subssytem name.
    
    std::string GetSubSystem (void) {
        return I_SubSystem;
    }
    
    //  SetLevelNames() takes a comma-separated list of all the levels used by this subsystem.
    
    void SetLevelNames (const std::string& List) {
        TcsUtil::Tokenize(List,I_Levels,",");
        I_Flags.resize(I_Levels.size());
        for (int& Flag : I_Flags) { Flag = false; }
    }
    
    //  ListLevels() returns the level names used by this subsystem as a comma-separated string.
    
    std::string ListLevels (void) {
        bool First = true;
        std::string Levels = "";
        for (std::string& Level : I_Levels) {
            if (!First) Levels = Levels + ",";
            First = false;
            Levels = Levels + Level;
        }
        return Levels;
    }
    
    //  EnableLevels() enables any levels that match the list it is passed.
    
    void EnableLevels (const std::string& Levels) {
        (void) SetUnsetLevels (Levels,true);
    }

    //  DisableLevels() disables any levels that match the list it is passed.

    void DisableLevels (const std::string& Levels) {
        (void) SetUnsetLevels (Levels,false);
    }
    
    //  CheckLevels() checks a list of levels and returns a comma-separated
    //  list of those that it does not recognise.
    
    std::string CheckLevels (const std::string& Levels) {
        return SetUnsetLevels (Levels,true,true);
    }

    //  Active() returns true if the named level is currently active.
    
    bool Active (const std::string& Level) {
        bool Match = false;
        int NLevels = I_Levels.size();
        for (int I = 0; I < NLevels; I++) {
            if (TcsUtil::MatchCaseBlind(I_Levels[I].c_str(),Level.c_str())) {
                Match = I_Flags[I];
                break;
            }
        }
        return Match;
    }
    
    //  Log() outputs the text string supplied if the specified level is active.
    
    void Log (const std::string& Level,const std::string Text) {
        if (Active(Level)) {
            if (I_SubSystem != "") {
                printf ("[%s.%s] %s\n",I_SubSystem.c_str(),Level.c_str(),Text.c_str());
            } else {
                printf ("[%s] %s\n",Level.c_str(),Text.c_str());
            }
        }
    }

    //  Logf() is like Log() but provides printf() style formatting.

    void Logf (const std::string& Level,const char * const Format, ...) {
        if (Active(Level)) {
            char Message[1024];
            va_list Args;
            va_start (Args,Format);
            vsnprintf (Message,sizeof(Message),Format,Args);
            if (I_SubSystem != "") {
                printf ("[%s.%s] %s\n",I_SubSystem.c_str(),Level.c_str(),Message);
            } else {
                printf ("[%s] %s\n",Level.c_str(),Message);
            }
        }
    }

    //  Deprecated routine names.
    
    void LevelsList (const std::string& List) {
        SetLevelNames(List);
    }
    void SetLevels (const std::string& Levels) {
        EnableLevels (Levels);
    }
    void UnsetLevels (const std::string& Levels) {
        DisableLevels (Levels);
    }


private:
    
    //  SetUnsetLevels() does all the work for both EnableLevels() and
    //  DisableLevels(), the only difference being whether the matching levels
    //  are activated or deactivated.
    //
    //  Levels a list of level specifiers, comma-separated, with each specifier
    //         a string that can include wildcard characters. This routine
    //         finds all matching levels and activates/deactivates them.
    //  Set    true if the matching levels are to be made active, false if
    //         they are to be deactivated.
    //  Check  if true, levels are not modified. The intent is that this mode
    //         can be used simply to verify a levels specification.
    //
    //  Returns: A comma-separated string giving the unrecognised levels.
    
    std::string SetUnsetLevels (const std::string& Levels, bool Set, bool Check = false) {
        
        std::string Unrecognised = "";
        
        //  Split the Levels string into comma-separated tokens.
        
        std::vector<std::string> Tokens;
        TcsUtil::Tokenize(Levels,Tokens,",");
        
        //  Work through the tokens (each should be 'subsystem,level') one by one.
        
        for (std::string Item : Tokens) {
            
            bool Known = false;
            
            //  Check for negation using '!' and reverse the effect of Set if present.
            
            bool Enable = Set;
            if (Item.size() > 0 && Item[0] == '!') {
                Item = Item.substr(1);
                Enable = !Set;
            }
            
            //  Split the token into subsystem and level. Defaulting to '*' means a missing
            //  subsystem or level spec applies to all subsystems or levels.
            
            std::string SubSystem = "*";
            std::string Level = "*";
            size_t Dot = Item.find('.');
            if (Dot == std::string::npos) {
                Level = Item;
            } else {
                SubSystem = Item.substr(0,Dot);
                if (Dot < Item.size()) {
                    Level = Item.substr(Dot + 1);
                }
            }
            
            //  If the subsystem matches ours, check the level against all our levels.
            //  Enable or disable any that match.
            
            if (WildcardMatchCaseBlind(SubSystem.c_str(),I_SubSystem.c_str())) {
                int NLevels = I_Levels.size();
                for (int I = 0; I < NLevels; I++) {
                    if (WildcardMatchCaseBlind(Level.c_str(),I_Levels[I].c_str())) {
                        I_Flags[I] = Enable;
                        Known = true;
                    }
                }
            }
            
            if (!Known) {
                if (Unrecognised == "") Unrecognised = Item;
                else Unrecognised += "," + Item;
            }
        }
        return Unrecognised;
    }
    
    //  The name of the current sub-system.
    std::string I_SubSystem;
    //  All the individual level names.
    std::vector<std::string> I_Levels;
    //  Flags for each level, true when the level is active.
    std::vector<int> I_Flags;
};

#endif

// -------------------------------------------------------------------------------------------------

/*                        P r o g r a m m i n g  N o t e s

 o  I did play with using a map<string,bool> instead of the two vectors, one
    for the strings and one for the flags, but found it too awkward in the end,
    althogh it does feel like the obvious implementation. Maybe I'm just not
    as au fait with maps as I should be. I tried having I_Flags as a
    vector<bool> but this was made awkward by the speciailised implementation
    of vector<bool> which potentially packs up individual bools into bit
    patterns for efficiency. Storage efficiency isn't really important here.
 */
