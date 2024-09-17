//  ----------------------------------------------------------------------------
//
//                    C o m m a n d  H a n d l e r . c p p
//
//  An initial version of code implementing a general purpose command-line
//  parser. At the moment, the comments are pretty sparse and erratic, as
//  the structure is still evolving and I don't want to put too much work '
//  into comments until I'm hapy with the overall structure. At the moment,
//  this will handle a simple command line with various named parameters,
//  which is all that's needed for some programs, but it still doesn't have
//  the helpful prompting for unspecified parameters that I'd really like
//  to have working. Still a work in progress.
//
//  Author: Keith Shortridge, K&V (Keith@KnaveAndVarlet.com.au)
//
//  History:
//     18th Dec 2020. First useful version. Handles a simple command line,
//                    but no support for prompting for missing parameters. KS.
//     24th Feb 2021. Substantial reworking, to use strings internally
//                    to hold all values, moving almost all the code in
//                    the individual inheriting classes into the base CmdArg
//                    class. KS.
//     11th Jun 2024. Corrected (after some time!) a careless use of a bitwise
//                    operator in FileArg::AllowedValue(). KS.
//     20th Jun 2024. Minor change so the command handler will no longer mis-
//                    identify a positional value as specifying a parameter
//                    already specified explicity by name. KS.
//     25th Jul 2024. Added GetRange() to RealArg and IntArg. Mostly to help
//                    with argument types derived from these. Brought comments
//                    for FileArg up to date. FileArg now allows the case
//                    where NullOK and MustExist are both specified and the
//                    value given is a null string. KS.
//      2nd Aug 2024. Removed an unnecessary include of uuid/uuid.h, and fixed
//                    a warning from g++ about a catch statement. KS.
//      9th Aug 2024. Now supports an exit request (a "!" from a prompt), and
//                    a new built-in 'help' argument that lists the argument
//                    details (and requests an exit). Some reworking of error
//                    handling so the handler knows about any errors reported
//                    by the individual arguments. Strings set using SetText()
//                    are now displayed if the response to a prompt is '?'. KS.
//     15th Aug 2024. Now supports the use of a CmdArgHelper for arguments,
//                    and allows explicit null strings to be specified at a
//     19th Aug 2024. A '?' as the command line value for a parameter now
//                    forces a prompt for it, accompanied by help text. KS.
//                    prompt using "" or ''. Removed some unneeded code. KS.

#include "CommandHandler.h"

#include "TcsUtil.h"
#include "ReadFilename.h"

#include <stdio.h>
#include <unistd.h>
#include <sys/errno.h>
#include <string.h>
#include <glob.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <arpa/inet.h>
#include <pwd.h>
#include <sys/ioctl.h>

#include <iostream>
#include <fstream>

using std::string;
using std::list;
using std::vector;

//  ----------------------------------------------------------------------------
//
//                          C m d  A r g

CmdArg::CmdArg (CmdHandler& Handler,const string& Name,int Posn,
                 const string& Flags,const string& Prompt,const string& Text)
{
   I_Handler = &Handler;
   I_Name = Name;
   I_Prompt = Prompt;
   I_Text = Text;
   I_Posn = Posn;
   I_Helper = nullptr;
   I_PreviousSet = false;
   I_Previous = "";
   I_IsSet = false;
   I_SetValue = "";
   I_DefaultSet = false;
   I_Default = "";
   I_Reset = "";
   I_Value = "";
   I_IsActive = true;
   I_Listing = false;
   I_PromptOn = false;
   I_PromptOff = false;
   I_ArgType = "";
   I_ConstructError = "";
   I_ErrorText = "";
   
   //  Deal with any options specified in the Flags argument. If any aren't
   //  recognised, they may be argument-specific. Add them to the argument
   //  specific set for the argument-specific constructor to handle.
   
   I_Flags = 0;
   I_UnknownFlags.clear();
   vector<string> Tokens;
   TcsUtil::Tokenize(Flags,Tokens," ,");
   for (string& Token : Tokens) {
      if (TcsUtil::MatchCaseBlind(Token,"Required")) {
         I_Flags |= REQUIRED;
      } else if (TcsUtil::MatchCaseBlind(Token,"Valopt")) {
         I_Flags |= VALOPT;
      } else if (TcsUtil::MatchCaseBlind(Token,"Valreq")) {
         I_Flags |= VALREQ;
      } else if (TcsUtil::MatchCaseBlind(Token,"Hidden")) {
         I_Flags |= HIDDEN;
      } else if (TcsUtil::MatchCaseBlind(Token,"Nosave")) {
         I_Flags |= NOSAVE;
      } else if (TcsUtil::MatchCaseBlind(Token,"IsFile")) {
         I_Flags |= IS_FILE;
      } else if (TcsUtil::MatchCaseBlind(Token,"Internal")) {
         I_Flags |= INTERNAL;
      } else {
         I_UnknownFlags.push_back(Token);
      }
   }
   
   //  The default for most argument types is that a value is required.

   if (I_Flags & VALOPT) {
      I_Valopt = true;
   } else {
      I_Valopt = false;
   }

   if (I_Handler) I_Handler->AddArg(this);
}

CmdArg::~CmdArg()
{
   if (I_Handler) I_Handler->DelArg(this);
}

void CmdArg::CloseHandler (CmdHandler* Handler)
{
   if (I_Handler == Handler) {
      I_Handler = NULL;
      I_IsActive = false;
   }
}

void CmdArg::Reset (void)
{
   //  "Reset' essentially means ignore any previous values, and the easiest
   //  way to achieve that is to clear the flag that says a previous value
   //  was set.
   
   I_PreviousSet = false;
}

vector<string> CmdArg::GetUnknownFlags (void)
{
   return I_UnknownFlags;
}

void CmdArg::SetUnknownFlags (const vector<string>& Flags)
{
   I_UnknownFlags = Flags;
}

void CmdArg::SetListing (bool Listing)
{
   I_Listing = Listing;
}

void CmdArg::SetPrompt (const string& Prompt)
{
   I_Prompt = Prompt;
}

void CmdArg::SetText (const string& Text)
{
   I_Text = Text;
}

void CmdArg::SetPrompting (const string& OnOffDefault)
{
   if (TcsUtil::MatchCaseBlind(OnOffDefault,"On")) {
      I_PromptOn = true;
      I_PromptOff = false;
   } else if (TcsUtil::MatchCaseBlind(OnOffDefault,"Off")) {
      I_PromptOn = false;
      I_PromptOff = true;
   } else if (TcsUtil::MatchCaseBlind(OnOffDefault,"Default")) {
      I_PromptOn = false;
      I_PromptOff = false;
   }
}

string CmdArg::Description (void)
{
   string Text = "'" + I_Name + "'";
   if (I_Prompt != "") Text += " (" + I_Prompt + ")";
   return Text;
}

bool CmdArg::TellUser (const string& ReportString)
{
   bool ReturnOK = true;
   if (I_Handler) {
      CmdInteractor* Interactor = I_Handler->GetInteractor();
      if (Interactor) {
         ReturnOK = Interactor->WriteLn(ReportString);
      }
   }
   return ReturnOK;
}

bool CmdArg::ValidValue (const string& /*Value*/)
{
   // By default, any string is valid. Most arguments will need to override
   // this.
   
   return true;
}

bool CmdArg::AllowedValue (const string& /*Value*/)
{
   // By default, any string is allowed. Most arguments will need to override
   // this.
   
   return true;
}

bool CmdArg::MatchName (const string& Name)
{
   bool Match = false;
   if (PosName(Name)) {
      Match = true;
   } else if (Negatable() && NegName(Name)) {
      Match = true;
   }
   return Match;
}

bool CmdArg::NegName (const string& Name)
{
   bool Match = false;
   int Pos = 0;
   if (Name.substr(0,1) == "-") Pos = 1;
   if (TcsUtil::MatchCaseBlind(Name.substr(Pos),"no" + I_Name)) {
      Match = true;
   }
   return Match;
}

bool CmdArg::PosName (const string& Name)
{
   bool Match = false;
   int Pos = 0;
   if (Name.substr(0,1) == "-") Pos = 1;
   if (TcsUtil::MatchCaseBlind(Name.substr(Pos),I_Name)) {
      Match = true;
   }
   return Match;
}

void CmdArg::OutputHelpText(void)
{
   string HelpText = "";
   if (I_Helper) HelpText = I_Helper->HelpText();
   if (I_Text == "" && HelpText == "") {
      TellUser("Sorry, no detailed description available for '"
               + I_Name + "'.");
   } else {
      if (I_Text != "") TellUser(I_Text);
      if (HelpText != "") TellUser(HelpText);
   }
}

string CmdArg::PromptUser (const string& Default)
{
   bool OK = true;
   string Value = "";
   if (I_Handler) {
      CmdInteractor* Interactor = I_Handler->GetInteractor();
      if (Interactor) {
         string Prompt = Description() + " [" + Default + "] ? ";
         for (;;) {
            OK = Interactor->ReadLn (Value,Prompt,IsFile());
            if (!OK) {
               Value = "";
               break;
            } else {
               if (Value == "\'?\'" or (Value == "\"?\"")) Value = "?";
               if (Value == "?") {
                  OutputHelpText();
               } else {
                  break;
               }
            }
         }
      }
   }
   return Value;
}

bool CmdArg::CanPrompt (void)
{
   bool ReturnOK = false;
   if (I_Handler) {
      CmdInteractor* Interactor = I_Handler->GetInteractor();
      if (Interactor) {
         ReturnOK = Interactor->IsInteractive();
      }
   }
   return ReturnOK;
}

bool CmdArg::SetPrevious (const string& Value)
{
   //  We could test for validity, but it's better to just accept the
   //  value, valid or not, and let GetValue() do the checking when it
   //  comes to use it, since it has the option of interacting with the
   //  user if the value is invalid (and allowed value may have changed
   //  by the time GetValue() is called.
   
   bool ReturnOK = true;
   I_Previous = Value;
   I_PreviousSet = true;
   return ReturnOK;
}

void CmdArg::SetDefaultValue (const string& Default)
{
   //  The same considerations about validity checking apply as for
   //  SetPrevious().
   
   I_Default = Default;
   I_DefaultSet = true;
}

string CmdArg::DisplayValue (const string& Value)
{
   //  For most arguments, the strings used to store values are already
   //  in a suitable format to display to the user. Only arguments that store
   //  values in an internal format need to implement anything more complex.
   
   return Value;
}

string CmdArg::UnspecifiedValue(void)
{
   //  Should return the value to be used when an argument that doesn't
   //  require a value is specified without a value. For example, a boolean
   //  argument would normally default to 'true' (ie if 'prompt' is specified
   //  on the command line, this is taken a setting the 'prompt' argument to
   //  'true'. Most arguments don't need to worry about this.
   
   return "";
}

void CmdArg::SetError (const string& ErrorText)
{
   if (I_Handler) {
      I_ErrorText = ErrorText;
      I_Handler->SetError(ErrorText);
   }
}

bool CmdArg::SetValue (const string& Name, const string& Value)
{
   bool ReturnOK = true;
   
   //  Rather like with SetPrevious(), it's probably best not to try
   //  to validate the value here. Better just to record it and let
   //  GetValue() handle the validity checking. (Note: this is normally
   //  only called from the CommandHandler during command line parsing.)
   
   if (I_IsSet) {
      SetError("Cannot set argument " + Description() + " to " +
                    Value + " - already set to " + DisplayValue(I_SetValue));
      ReturnOK = false;
   } else {
      I_SetValue = Value;
      if (NegName(Name)) {
         if (Negatable()) {
            I_SetValue = NegateValue(I_SetValue);
         } else {
            ReturnOK = false;
            SetError (Description() +
                   " cannot be used in a negative form. '" + Name +
                                     "' is invalid");
         }
      }
      if (ReturnOK) I_IsSet = true;
   }
   return ReturnOK;
}

bool CmdArg::SetValue (const std::string& Name, bool Equals,
       const std::string Value, bool GotValue, bool* UsedValue)
{
   bool ReturnOK = true;
   *UsedValue = false;
   //  We can assume the name matches - but is it the -ve version?
   if (Equals) {
      ReturnOK = SetValue(Name,Value);
      *UsedValue = ReturnOK;
   } else {
      bool Used = false;
      if (GotValue) {
         //  '?' is always valid, otherwise check before accepting.
         if (Value == "?" || ValidValue(Value)) {
            Used = SetValue(Name,Value);
               if (Used) {
               *UsedValue = true;
            }
         }
      }
      if (!Used) {
         //  No value, is the name of the option enough? (If a value isn't
         //  optional, this shouldn't be called without a value, but we
         //  should check.)
         if (!ValueOptional()) {
            ReturnOK = false;
            SetError (Description() + " requires a value to be specified.");
         } else {
            I_SetValue = UnspecifiedValue();
            if (NegName(Name)) I_SetValue = NegateValue(I_SetValue);
            I_IsSet = true;
         }
      }
   }
   return ReturnOK;
}

string CmdArg::FormatForRecord (const string& Value)
{
   //  This routine is only used for output of the current values to
   //  the file that records them for use the next time the program is
   //  run. This means that they need to be capable of being parsed as
   //  part of a string containing argument names and values, and that
   //  means any strings with blanks in them need to be enclosed with
   //  quotes, as much completely null strings.
   
   string Text;
   if (I_Value.find(' ') != string::npos) {
      Text = '\"' + Value + '\"';
   } else {
      if (Value == "") {
         Text = "\"\"";
      } else {
         Text = Value;
      }
   }
   return Text;
}

string CmdArg::NegateValue(const std::string& Value)
{
   //  Most arguments won't support negation, so this routine can be null.
   //  Any that do, need to override this routine with one that takes a
   //  string value for the argument and returns a string representing its
   //  negation. A boolean argument will take "false" and return "true",
   //  for example.
   
   return Value;
}

string CmdArg::Requirement (void)
{
   //  Needs to return a description of any requirements for the argument.
   //  This is very much up to the argument itself, and may even change
   //  dynamically - for example, a numeric value with limits that may
   //  change. This returns a very general fallback string.
   
   return "any valid string";
}

void CmdArg::ExpandValue (std::string* Value)
{
   //  Provides a place for an argument to modify a supplied value in some
   //  argument-specific way. Most arguments don't do anything, which is
   //  which Value is passed as an address that can be left unchanged
   //  quite efficiently rather than replacing it with itself as the
   //  return value. A file argument may want to expand any environment
   //  variables. A numeric argument may want to interpret options such as
   //  'min' or 'max'.
}

string CmdArg::GetCurrentSpec (void)
{
   string Spec = "";
   if (!Internal()) {
      if (ValidValue(I_Value)) {
         Spec = I_Name + " = " + FormatForRecord(I_Value);
      }
   }
   return Spec;
}

bool CmdArg::Clear (void)
{
   //  Initialises ready for a new parse pass through a new set of argument
   //  strings. This is also a place to check on whether there are any
   //  so far unreported arrors - particularly, ones occurring during the
   //  constructor, which would otherwise go unreported.
   
   bool ReturnOK = true;
   I_IsSet = false;
   I_Listing = false;
   if (I_ConstructError != "") {
      SetError(I_ConstructError);
      ReturnOK = false;
   } else {
      int Count = 0;
      for (string& Flag : I_UnknownFlags) {
         string ErrorText = "";
         if (Count++ == 0) {
            ErrorText = "Unrecognised flags in constructor for "
                                                  + I_Name + ": " + Flag;
         } else {
            ErrorText += "," + Flag;
         }
         SetError(ErrorText);
      }
      if (Count > 0) ReturnOK = false;
   }
   return ReturnOK;
}

string CmdArg::GetArgType (void)
{
   return I_ArgType;
}

void CmdArg::SetArgType (const string& Type)
{
   I_ArgType = Type;
}

void CmdArg::SetHelper (CmdArgHelper* Helper)
{
   I_Helper = Helper;
}

string CmdArg::GetError (void)
{
   return I_ErrorText;
}

bool CmdArg::DetermineValue (string* CurrentValue)
{
   //  Sets Value to the current value of the argument, based on any
   //  argument parsing and on possible intraction with the user. This is
   //  what does all the work for those GetValue() calls in the derived
   //  classes.
   
   bool ReturnOK = true;
   string Value = "";
   string Origin;
   bool Forced = false;
   bool WasPrompted = false;
   if (I_Handler && !I_Handler->AllOK()) {
      I_ErrorText = I_Handler->GetError();
      ReturnOK = false;
   }
   if (ReturnOK) {
      if (I_IsSet) {
         //  We already have a value from the parsing. Check for a request
         //  for details, ie a '?' entered on the command line.
         if (I_SetValue == "?") {
            OutputHelpText();
            I_IsSet = false;
            Forced = true;
         } else {
            //  Otherwise, see if the supplied value is valid.
            Origin = "Specified";
            Value = I_SetValue;
            ExpandValue (&Value);
            if (!AllowedValue(Value)) {
               TellUser (Origin + " value ('" + DisplayValue(Value) + "') for "
                           + I_Name + " is invalid. Must be " + Requirement());
               I_IsSet = false;
               Forced = true;
            } else {
               if (I_Helper) {
                  string Reason;
                  if (!I_Helper->CheckValidity(Value,&Reason)) {
                     TellUser (Origin + " value ('" + DisplayValue(Value) +
                               "') for " + I_Name + " is invalid. " + Reason);
                     I_IsSet = false;
                     Forced = true;
                  }
               }
            }
         }
      }
      if (!I_IsSet) {
         //  Set a default value and note where it came from.
         string Default,DefOrigin;
         if (I_DefaultSet) {
            Default = I_Default; DefOrigin = "Default";
         } else if (I_PreviousSet) {
            Default = I_Previous; DefOrigin = "Previous";
         } else {
            Default = I_Reset; DefOrigin = "Reset";
         }
         //  If the default is invalid, we're forced to prompt.
         ExpandValue (&Default);
         bool DefaultValid = AllowedValue(Default);
         if (!DefaultValid) {
            Forced = true;
         } else if (I_Helper) {
            string Reason;
            if (!I_Helper->CheckValidity(Value,&Reason)) Forced = true;
         }
         string DefString = DisplayValue(Default);
         if (CanPrompt()) {
            //  We can prompt, do we have to?
            bool Prompt =
                 ((Required() || I_PromptOn) && !I_PromptOff) || Forced;
            if (Prompt) {
               //  Continue prompting until we get a valid response.
               for (;;) {
                  string Reply = PromptUser (DefString);
                  //  A null response accepts the default, but we allow a
                  //  null string to be given explicitly as "" or ''.
                  if (Reply == "") Reply = Default;
                  if (Reply == "\"\"" || Reply == "\'\'") Reply = "";
                  //  And '!' is an exit request. But '!' or "!' means !
                  if (Reply == "!") {
                     ReturnOK = false;
                     if (I_Handler) I_Handler->RequestExit();
                     break;
                  }
                  if (Reply == "\'!\'" or (Reply == "\"!\"")) Reply = "!";
                  ExpandValue (&Reply);
                  //  Was this a valid reply?
                  bool Valid = ValidValue(Reply);
                  if (Valid) Valid = AllowedValue(Reply);
                  if (Valid && I_Helper) {
                     string Reason;
                     if (!I_Helper->CheckValidity(Reply,&Reason)) {
                        TellUser (Reason);
                        Valid = false;
                     }
                  }
                  if (Valid) {
                     WasPrompted = true;
                     Origin = "Prompted";
                     Value = Reply;
                     break;
                  } else {
                     TellUser("The argument value must be " + Requirement());
                  }
               }
            } else {
               //  The value wasn't specified, but we have a valid default,
               //  and we weren't forced to prompt for it.
               Value = Default;
               Origin = DefOrigin;
            }
         } else {
            //  The value wasn't specified, and we weren't able to prompt.
            if (Required()) {
               SetError("No value provided for required argument: "
                                                               + Description());
               ReturnOK = false;
            } else {
               if (DefaultValid) {
                  Value = Default;
                  Origin = DefOrigin;
               } else {
                  //  We wanted to force prompting because we didn't have a
                  //  valid value, but we can't prompt, so we error.
                  SetError(DefOrigin + " ('" + DefString + "') for "
                         + I_Name + " is invalid. Must be " + Requirement());
                  ReturnOK = false;
               }
            }
         }
      }
   }
   if (ReturnOK) {
      ExpandValue(&Value);
      if (I_Listing) {
         if (!WasPrompted) {
            string ReportString = Description() + " = " + DisplayValue(Value);
            TellUser (ReportString);
         }
      }
      I_Value = Value;
   }
   *CurrentValue = I_Value;
   return ReturnOK;
}

//  ----------------------------------------------------------------------------
//
//                          S t r i n g  A r g

StringArg::StringArg (CmdHandler& Handler,const string& Name,int Posn,
   const string& Flags,const string& Reset,const string& Prompt,
   const string& Text) :
      CmdArg (Handler,Name,Posn,Flags,Prompt,Text)
{
   SetReset(Reset);
   if (ValOpt()) {
      SetConstructError("String arguments like '" + GetName() +
      "' cannot have optional arguments - remove VALOPT flag from constructor");
   }
   SetValOpt(false);
   SetArgType("string");
}

StringArg::~StringArg()
{
}

string StringArg::GetValue (bool* Ok,string* Error)
{
   string Value = "";
   if (*Ok) {
      if (!DetermineValue(&Value)) {
         *Ok = false;
         *Error = GetError();
      }
   }
   return Value;
}

void StringArg::SetDefault (const std::string& Default)
{
   SetDefaultValue(Default);
}

//  ----------------------------------------------------------------------------
//
//                          F i l e  A r g
//
//  This code treats a FileArg just the same as a StringArg, but it adds
//  support for checking to see the name parses properly, for optionally
//  checking for an existing file, for a set of possible 'reset' values,
//  and for filename completion when prompting.

FileArg::FileArg (CmdHandler& Handler,const string& Name,int Posn,
      const string& Flags,const string& Reset,const string& Prompt,
      const string& Text) :
      StringArg (Handler,Name,Posn,Flags,Reset,Prompt,Text)
{
   I_FileFlags = 0;
   
   //  Check any Flags that the base class did not recognise. Pass back any
   //  that remain unhandled (and the base class will handle the messaging).
   
   vector<string> UnknownFlags = GetUnknownFlags();
   vector<string> StillUnknown;
   for (string& Flag : UnknownFlags) {
      if (TcsUtil::MatchCaseBlind(Flag,"MustExist")) {
         I_FileFlags |= MUST_EXIST;
      } else if (TcsUtil::MatchCaseBlind(Flag,"NullOk")) {
         I_FileFlags |= NULL_OK;
      } else {
         StillUnknown.push_back(Flag);
      }
   }
   SetUnknownFlags(StillUnknown);
   SetArgType("file");

   //  A File argument may have a reset value which is a list of possible
   //  defaults, acting as a search path. Pick the first one that is an
   //  acceptable option. Note that even if there is only one option, this
   //  code checks to see if it can be expanded, and if so makes the expanded
   //  version the default.
   
   TcsUtil::Tokenize(Reset,I_ResetValues," ,");
   if (I_ResetValues.size() >= 1) {
      bool Set = false;
      for (string& Res : I_ResetValues) {
         string Expanded;
         if (TcsUtil::ExpandFileName(Res,Expanded)) {
            if (MustExist() && !FileExists(Expanded)) continue;
            SetReset(Res);
            Set = true;
            break;
         }
      }
      
      //  If none are acceptable, make the default the last one - the final
      //  fallback. The important thing is only to have one reset value.
      
      if (!Set) SetReset(I_ResetValues[I_ResetValues.size() - 1]);
   }

   //  Make sure the IS_FILE flag is set

   SetFlag(IS_FILE);

}

FileArg::~FileArg()
{
}

bool FileArg::FileExists (const string& FileName)
{
   return (access(FileName.c_str(), F_OK ) != -1);
}

bool FileArg::AllowedValue (const string& Value)
{
   bool Allowed = true;
   if (Value == "") {
      if (!NullOk()) Allowed = false;
   } else {
      if (MustExist() && !FileExists(Value)) Allowed = false;
   }
   return Allowed;
}

string FileArg::Requirement (void)
{
   string Text;
   if (MustExist()) {
      Text = "the name of an existing file";
   } else {
      Text = "a valid file name";
   }
   return Text;
}

void FileArg::ExpandValue (string* Value)
{
   string ExpandedValue;
   if (TcsUtil::ExpandFileName(*Value,ExpandedValue)) {
      ExpandTildePath(&ExpandedValue);
      if (*Value != ExpandedValue) {
         if (IsListing()) {
            TellUser ("Note: '" + *Value + "' expands to '" +
                                    ExpandedValue + "'");
         }
         *Value = ExpandedValue;
      }
   }
}

void FileArg::ExpandTildePath (string* Path)
{
   if (Path->size() > 1) {
      if ((*Path)[0] == '~') {
         size_t Idx = Path->find('/');
         if (Idx != string::npos) {
            string TildePart = Path->substr(0,Idx);
            glob_t HomeGlob;
            if (glob(TildePart.c_str(),GLOB_TILDE,NULL,&HomeGlob) == 0) {
               if (HomeGlob.gl_pathc == 1) {
                  string ExpandedPath = string(HomeGlob.gl_pathv[0]) +
                                                      Path->substr(Idx);
                  *Path = ExpandedPath;
               }
            }
         }
      }
   }
}


//  ----------------------------------------------------------------------------
//
//                          B o o l  A r g

BoolArg::BoolArg (CmdHandler& Handler,const string& Name,int Posn,
     const string& Flags,bool Reset,const string& Prompt,const string& Text) :
     CmdArg (Handler,Name,Posn,Flags,Prompt,Text)
{
   SetReset(Reset ? "true" : "false");
   
   //  Most argument types default to requiring a value unless 'value optional'
   //  is explicitly specified in Flags, and the CmdArg constructor will have
   //  done that for us already. The default for a boolean argument is that a
   //  value is optional, and unless 'value required' is explicitly' specified,
   //  that's what we set here.
   
   if (!ValReq()) SetValOpt(true);
   
   //  All Boolean arguments are negatable. Maybe there's a case for having
   //  a flag that controls this, for any odd case where someone wants
   //  non-negatable boolean argument?
   
   SetFlag(NEGATABLE);
   
   SetArgType("bool");
}

BoolArg::~BoolArg()
{
}

bool BoolArg::CheckValidValue (const string& Value, bool* BoolValue)
{
   bool Valid = false;
   *BoolValue = false;
   int Len = Value.size();
   if (Len > 0) {
      Valid = true;
      if (TcsUtil::MatchCaseBlind(Value,string("Yes").substr(0,Len))) {
         *BoolValue = true;
      } else if (TcsUtil::MatchCaseBlind(Value,string("True").substr(0,Len))) {
         *BoolValue = true;
      } else if (TcsUtil::MatchCaseBlind(Value,string("No").substr(0,Len))) {
         *BoolValue = false;
      } else if (TcsUtil::MatchCaseBlind(Value,string("False").substr(0,Len))) {
         *BoolValue = false;
      } else {
         Valid = false;
      }
   }
   return Valid;
}

bool BoolArg::ValidValue (const string& Value)
{
   bool BoolValue;
   return CheckValidValue (Value,&BoolValue);
}

bool BoolArg::AllowedValue (const string& Value)
{
   return ValidValue(Value);
}

string BoolArg::NegateValue(const std::string& Value)
{
   string NegValue = "";
   bool BoolValue;
   if (CheckValidValue(Value,&BoolValue)) {
      NegValue = BoolValue ? "false" : "true";
   }
   return NegValue;
}

bool BoolArg::GetValue (bool* Ok,string* Error)
{
   bool BoolValue = false;
   if (*Ok) {
      string Value;
      if (DetermineValue(&Value)) {
         (void) CheckValidValue(Value,&BoolValue);
      } else {
         *Ok = false;
         *Error = GetError();
      }
   }
   return BoolValue;
}

string BoolArg::Requirement (void)
{
   //  Returns a string describing the requirements for a valid value.
   
   return "a string that is one of 'true','false','yes' or 'no'";
}

string BoolArg::UnspecifiedValue(void)
{
   //  Should return the value to be used when an argument that doesn't
   //  require a value is specified without a value. For example, a boolean
   //  argument would normally default to 'true' (ie if 'prompt' is specified
   //  on the command line, this is taken a setting the 'prompt' argument to
   //  'true'. Most arguments don't need to worry about this.
   
   return "true";
}

void BoolArg::SetDefault (bool Default)
{
   SetDefaultValue(Default ? "true" : "false");
}


//  ----------------------------------------------------------------------------
//
//                          I n t  A r g

IntArg::IntArg (CmdHandler& Handler,const string& Name,int Posn,
      const string& Flags,long Reset,long Min,long Max,
      const string& Prompt,const string& Text) :
         CmdArg (Handler,Name,Posn,Flags,Prompt,Text)
{
   SetReset(FormatInt(Reset));
   I_Min = Min;
   I_Max = Max;
   SetArgType("int");
}

IntArg::~IntArg()
{
}

long IntArg::GetValue (bool* Ok,string* Error)
{
   long IntValue = 0;
   if (*Ok) {
      string Value;
      if (DetermineValue(&Value)) {
         (void) CheckValidValue(Value,&IntValue);
      } else {
         *Ok = false;
         *Error = GetError();
      }
   }
   return IntValue;
}

void IntArg::SetRange (long Min, long Max)
{
    I_Min = Min;
    I_Max = Max;
}

void IntArg::GetRange (long* Min, long* Max)
{
    *Min = I_Min;
    *Max = I_Max;
}

bool IntArg::CheckValidValue (const string& String, long* Value)
{
   bool Valid = true;
   size_t Idx;
   try {
      *Value = stol(String,&Idx);
   } catch (...) {
      *Value = 0;
      Valid = false;
   }
   if (Valid) {
      if ((Idx != string::npos) && (Idx != String.size())) Valid = false;
   }
   return Valid;
}

bool IntArg::ValidValue (const string& Value)
{
   long IntValue;
   return (CheckValidValue(Value,&IntValue));
}

bool IntArg::AllowedValue (const string& Value)
{
   bool Valid = false;
   long IntValue;
   if (CheckValidValue(Value,&IntValue)) {
      if (I_Min == I_Max) {
         Valid = true;
      } else {
         Valid = (IntValue >= I_Min && IntValue <= I_Max);
      }
   }
   return Valid;
}

string IntArg::Requirement (void)
{
   string Text = "an integer in the range " + FormatInt(I_Min) + " to " +
                                                          FormatInt(I_Max);
   return Text;
}

string IntArg::FormatInt (long Value)
{
   char Number[64];
   snprintf(Number,sizeof(Number),"%ld",Value);
   return string(Number);
}

void IntArg::SetDefault (long Default)
{
   SetDefaultValue(FormatInt(Default));
}

void IntArg::ExpandValue (string* Value)
{
   string ExpandedValue;
   bool Changed = false;
   if (TcsUtil::MatchCaseBlind(*Value,"Max")) {
      ExpandedValue = FormatInt(I_Max);
      Changed = true;
   } else if (TcsUtil::MatchCaseBlind(*Value,"Min")) {
      ExpandedValue = FormatInt(I_Min);
      Changed = true;
   }
   if (Changed) {
      if (IsListing()) {
         TellUser ("Note: '" + *Value + "' expands to '" +
                                    ExpandedValue + "'");
      }
      *Value = ExpandedValue;
   }
}


//  ----------------------------------------------------------------------------
//
//                          R e a l  A r g

RealArg::RealArg (CmdHandler& Handler,const string& Name,int Posn,
      const string& Flags,double Reset,double Min,double Max,
      const string& Prompt,const string& Text) :
         CmdArg (Handler,Name,Posn,Flags,Prompt,Text)
{
   SetReset(FormatRealExact(Reset));
   I_Min = Min;
   I_Max = Max;
   SetArgType("real");
}

RealArg::~RealArg()
{
}

double RealArg::GetValue (bool* Ok,string* Error)
{
   double RealValue = 0.0;
   if (*Ok) {
      string Value;
      if (DetermineValue(&Value)) {
         (void) CheckValidValue(Value,&RealValue);
      } else {
         *Ok = false;
         *Error = GetError();
      }
   }
   return RealValue;
}


void RealArg::SetRange (double Min, double Max)
{
    I_Min = Min;
    I_Max = Max;
}

void RealArg::GetRange (double* Min, double* Max)
{
    *Min = I_Min;
    *Max = I_Max;
}

bool RealArg::CheckValidValue (const string& String, double* Value)
{
   //  First see if this can be interpreted as a conventionally formatted
   //  value.
   
   bool Valid = true;
   size_t Idx;
   try {
      *Value = stod(String,&Idx);
   } catch (...) {
      Valid = false;
   }
   if (Valid) {
      if ((Idx != string::npos) && (Idx != String.size())) Valid = false;
   }
   
   //  If that fails, see if this is an 'exact' representation of the
   //  internal format of a double, created by FormatRealExact().
   
   if (!Valid) {
      Valid = ReadRealExact(String,Value);
   }
   return Valid;
}

bool RealArg::ValidValue (const string& Value)
{
   double RealValue;
   return (CheckValidValue(Value,&RealValue));
}

bool RealArg::AllowedValue (const string& Value)
{
   bool Valid = false;
   double RealValue;
   if (CheckValidValue(Value,&RealValue)) {
      if (I_Min == I_Max) {
         Valid = true;
      } else {
         Valid = (RealValue >= I_Min && RealValue <= I_Max);
      }
   }
   return Valid;
}

string RealArg::Requirement (void)
{
   string Text = "a floating point number in the range " +
                    FormatReal(I_Min) + " to " + FormatReal(I_Max);
   return Text;
}

string RealArg::FormatReal (double Value)
{
   char Number[64];
   snprintf(Number,sizeof(Number),"%g",Value);
   return string(Number);
}

string RealArg::FormatRealExact (double Value)
{
   //  Formats a value into to a string that can be output and read by
   //  ReadRealExact() with the guarantee that the double precision value
   //  that results will be the exact same value as the original value
   //  passed to this routine (at least, on the same machine or one that
   //  uses the same floating point representation.)
   //
   //  It does this by writing out not a conventionally formatted string,
   //  but a string of hex characters that exactly reproduce the internal
   //  floating point representation used by the machine this runs on. It
   //  embeds this in enough information about the format that - in principle
   //  - a sufficiently clever implementation of ReadRealExact() could
   //  re-interpret the number even on a machine that used a different
   //  floating point representation (although the 'exact' match might not
   //  be guaranteed then).
   //
   //  The format used is something like:
   //  <FE:08:L:d7f58faee7d64640=45.6789>
   //  Here the <> delimit the formatted value. 'F' indicates floating
   //  point, and 'E' indicates IEEE floating point format. The '08' are
   //  two hex digits giving the number of bytes used for the value
   //  internally. The 'L' (as opposed to 'B') indicates this is a little-
   //  endian machine. The 'd7f58faee7d64640' is the internal bit pattern
   //  used by this machine to represent the number in question, and to
   //  help humans read this, the '=45.6789' is an approximate conventionally
   //  formated respresentation (using %g) of the number itself (originally
   //  coded in the test program that generated this as 45.67894537.)
   
   unsigned char Hex[16] = {'0','1','2','3','4','5','6','7','8','9',
                                           'a','b','c','d','e','f'};
   double LocalValue = Value;
   int Len = sizeof(LocalValue);
   string Number;
   Number.reserve(64);
   
   //  Write the initial parts of the format into Number.
   
   char Endian = (htons(42) == 42) ? 'B' : 'L';
   Number = "<FE:";
   Number.push_back(Hex[(Len >> 4) % 0xf]);
   Number.push_back(Hex[Len & 0xf]);
   Number.push_back(':');
   Number.push_back(Endian);
   Number.push_back(':');
   
   //  Now work through the bytes of LocalValue one by one, encoding them
   //  into hex pairs. The elements of Fptr[] are in fact the bytes of
   //  LocalValue.
   
   unsigned char* Fptr = (unsigned char*)&LocalValue;
   for (int I = 0; I < Len; I++) {
      Number.push_back(Hex[Fptr[I] >> 4]);
      Number.push_back(Hex[Fptr[I] & 0xf]);
   }
   
   //  Finally, append a conventionally formatted version.
   
   Number += "=" + FormatReal(Value) + ">";
   return Number;
}

bool RealArg::HexCharChar2Char (char Ch1,char Ch2,unsigned char* Ch)
{
   //  This is a utility routine used by ReadRealExact(). There are a
   //  couple of places where that routine needs to convert two successive
   //  hexadecimal characters into the unsigned byte value they represent.
   //  That is what this does. For example, HexCharChar2Char('a','8',&Ch)
   //  will return Ch set to 0xa8. (If you're game to assume a character
   //  encoding, eg ASCII, then this can be done more simply, but this
   //  actually compares the characters with actual characters in the
   //  Hex[] array.) If the characters passed aren't valid Hex characters
   //  (they can be upper or lower case) then this routine returns false.
   //  And there ar eother ways of doing this with some more recent C++
   //  library routines, but they usually involve awkward substr() calls
   //  and having to check exceptions. I think this is simpler.
   
   char Hex[16] = {'0','1','2','3','4','5','6','7','8','9',
                                           'a','b','c','d','e','f'};
   *Ch = 0;
   bool Valid = false;
   char Ch1Low = tolower(Ch1);
   for (unsigned int N = 0; N < 16; N++) {
      if (Ch1Low == Hex[N]) {
         *Ch += N << 4;
         Valid = true;
         break;
      }
   }
   if (Valid) {
      Valid = false;
      char Ch2Low = tolower(Ch2);
      for (unsigned int N = 0; N < 16; N++) {
         if (Ch2Low == Hex[N]) {
            *Ch += N;
            Valid = true;
            break;
         }
      }
   }
   return Valid;
}

bool RealArg::ReadRealExact (const string& Value,double* RealValue)
{
   //  Look at the comments to FormatRealExact() for details of the
   //  format used. This code simply works tediously through all the
   //  sections of the format checking them for correctness, and
   //  finally decodes the internal representation of the floating
   //  point number. It doesn't try to handle numbers formatted by a
   //  machine that used a different floating point format, had a different
   //  'endian' layout, or a different number of bytes for a 'double'.
   //  In practice, at least in the context of a CommandHandler, this
   //  routine is unlikely to be run on anything other than the machine
   //  that originally encoded the string in question.
   
   bool Valid = false;
   *RealValue = 0.0;
   int Nch = Value.size();
   int MinLen = 11 + 2 * sizeof(double);
   if (Nch >= MinLen) {
      if (Value.substr(0,4) == "<FE:") {
      
         //  Get the two digits giving the double length in bytes
         //  and decode them and check they match sizeof(double).
         
         char Ch1 = Value[4];
         char Ch2 = Value[5];
         unsigned char Ch;
         if (HexCharChar2Char(Ch1,Ch2,&Ch)) {
            unsigned int Len = Ch;
            if (Len == sizeof(double)) {
            
               //  Check we have the same 'endian' setting.
               
               if (Value[6] == ':') {
                  char Endian = (htons(42) == 42) ? 'B' : 'L';
                  if (Value[7] == Endian && Value[8] == ':') {
                  
                     //  Finally, decode the hex representation of the
                     //  internal representation of the number. This is
                     //  just the reverse of what FormatRealExact() does,
                     //  treating LocalValue as a sequence of bytes and
                     //  getting their value one by one - writing to elements
                     //  of Fptr[] is actually writing to the bytes of
                     //  LocalValue. The value of each byte is given by
                     //  two successive hex digits.
                     
                     double LocalValue = 0;
                     unsigned char* Fptr = (unsigned char*)&LocalValue;
                     int Ich = 9;
                     Valid = true;
                     for (unsigned int I = 0; I < Len; I++) {
                        Ch1 = Value[Ich++];
                        Ch2 = Value[Ich++];
                        Ch = 0;
                        if (!HexCharChar2Char(Ch1,Ch2,&Ch)) {
                           Valid = false;
                           break;
                        } else {
                           Fptr[I] = Ch;
                        }
                     }
                     *RealValue = LocalValue;
                     
                     //  Check we're followed by an '=', but don't bother
                     //  with the conventionally formatted representation
                     //  that follows. (One could imagine checking that,
                     //  not for equality, of course, but for being close
                     //  enough given the number of significant figures.)
                     
                     if (Valid) {
                        if (Value[Ich] != '=') Valid = false;
                     }
                  }
               }
            }
         }
      }
   }
   return Valid;
}

string RealArg::DisplayValue (const string& Value)
{
   //  For most arguments, the strings as which values are stored are already
   //  in a suitable format to display to the user. Only arguments that store
   //  values in an internal format need to implement anything more complex.
   //  Real arguments need to do this.
   //
   //  The work here is done by CheckValidValue(), which can handle either a
   //  conventionally formatted real value, or one that represents the
   //  exact internal representation of a real value. It returns that value
   //  as a double, which FormatReal() will then format in the conventional
   //  way. (It would be more efficient to have a test that distinguished
   //  between the possible formats, in which case a conventionally formatted
   //  value could simply be returned as is, rather than reformatted.)
   
   double RealValue;
   string DisplayString = "";
   if (CheckValidValue(Value,&RealValue)) {
      DisplayString = FormatReal(RealValue);
   } else {
      DisplayString = Value;
   }
   return DisplayString;
}

void RealArg::SetDefault (double Default)
{
   SetDefaultValue(FormatRealExact(Default));
}

void RealArg::ExpandValue (string* Value)
{
   string ExpandedValue;
   bool Changed = false;
   if (TcsUtil::MatchCaseBlind(*Value,"Max")) {
      ExpandedValue = FormatRealExact(I_Max);
      Changed = true;
   } else if (TcsUtil::MatchCaseBlind(*Value,"Min")) {
      ExpandedValue = FormatRealExact(I_Min);
      Changed = true;
   }
   if (Changed) {
      if (IsListing()) {
         TellUser ("Note: '" + *Value + "' expands to '" +
                                 DisplayValue(ExpandedValue) + "'");
      }
      *Value = ExpandedValue;
   }
}

//  ----------------------------------------------------------------------------
//
//                          C m d  I n t e r a c t o r

CmdInteractor::CmdInteractor (void)
{
   I_IsInteractive = isatty(fileno(stdin));
   I_ErrorText = "";
   GetScreenWidth();
}

CmdInteractor::~CmdInteractor()
{
}

void CmdInteractor::SetInteractive (bool CanInteract)
{
   I_IsInteractive = CanInteract;
}

bool CmdInteractor::IsInteractive (void)
{
   return I_IsInteractive;
}

int CmdInteractor::ScreenWidth (void)
{
   //  Sometimes it's useful to know the screen width when formatting output.
   //  GetScreenWidth() returns what it thinks is the current screen width,
   //  or zero if it can't determine it. Note that this is the width when
   //  the program was launched. Trying to check on changes as the program
   //  runs gets very complex and doesn't seem warranted.
   
   int Width = 0;
   if (I_IsInteractive) Width = GetScreenWidth();
   return Width;
}

bool CmdInteractor::Write (const std::string& Text)
{
   std::cout << Text;
   return true;
}

bool CmdInteractor::WriteLn (const std::string& Text)
{
   std::cout << Text << std::endl;
   return true;
}

bool CmdInteractor::ReadLn (std::string& Text,
                               const std::string& Prompt, bool IsFile)
{
   bool ReturnOK = true;
   if (!I_IsInteractive) {
      ReturnOK = false;
      I_ErrorText = "Cannot read from user: program is not interactive.";
   } else {
      if (IsFile) {
         char Line[1024];
         if (ReadFilename (Prompt.c_str(),Line,sizeof(Line)) < 0) {
            I_ErrorText = "Error reading filename from user";
         } else {
            Text = string(Line);
         }
      } else {
         if (Prompt != "") Write(Prompt);
         try {
            std::getline (std::cin,Text);
         } catch (std::ifstream::failure& e) {
            I_ErrorText = "Error reading from user: " + string(e.what());
            ReturnOK = false;
         }
      }
   }
   return ReturnOK;
}

string CmdInteractor::GetError (void)
{
   return I_ErrorText;
}

int CmdInteractor::GetScreenWidth (void)
{
   //  GetScreenWidth() packages up the messy code needed by ScreenWidth() to
   //  get the actual width. Most modern Linux systems should support
   //  the winsize structure, but most example code around tests like this.
   
   int Width = 0;
#ifdef TIOCGWINSZ
   struct winsize WinSize;
   if (ioctl(fileno(stdin),TIOCGWINSZ,&WinSize) >= 0) Width = WinSize.ws_col;
#elif defined(TIOCGSIZE)
   struct ttysize WinSize;
   if (ioctl(fileno(stdin),TIOCGSIZE,&WinSize) >= 0) Width = WinSize.ts_cols;
#endif
   return Width;
}

//  ----------------------------------------------------------------------------
//
//                          C m d  H a n d l e r

CmdHandler::CmdHandler(const string& Program)
{
   I_Program = Program;
   I_Setup = false;
   I_ErrorText = "";
   I_ReadPrevious = false;
   I_ExitRequested = false;
   I_ErrorReported = false;
   I_ListArg = new BoolArg(*this,"List",0,"Internal,Hidden",
                                 false,"List values used for all arguments");
   I_PromptArg = new BoolArg(*this,"Prompt",0,"Internal,Hidden",
                             false,"Prompt for all arguments");
   I_HelpArg = new BoolArg(*this,"Help",0,"Internal,Hidden",
                             false,"List all arguments");
   I_ResetArg = new BoolArg(*this,
             "Reset",0,"Internal,Hidden",false,
                 "Reset values to default, ignoring previously used values");
   I_ExternalInteractor = false;
   I_Interactor = new CmdInteractor;
}

CmdHandler::~CmdHandler()
{
   for (CmdArg* Arg : I_CmdArgs) {
      if (Arg) Arg->CloseHandler(this);
   }
   if (I_ListArg) {
      I_ListArg->CloseHandler(this);
      delete I_ListArg;
   }
   if (I_PromptArg) {
      I_PromptArg->CloseHandler(this);
      delete I_PromptArg;
   }
   if (I_ResetArg) {
      I_ResetArg->CloseHandler(this);
      delete I_ResetArg;
   }
   if (I_Interactor && !I_ExternalInteractor) delete I_Interactor;
}

void CmdHandler::DelArg (CmdArg* Arg)
{
   I_CmdArgs.remove(Arg);
}

bool CmdHandler::AllOK (void)
{
   bool OK = true;
   if (I_ErrorReported || I_ExitRequested) OK = false;
   return OK;
}

void CmdHandler::SetInteractor (CmdInteractor* Interactor)
{
   if (I_Interactor && !I_ExternalInteractor) delete I_Interactor;
   I_Interactor = Interactor;
   I_ExternalInteractor = true;
}

bool CmdHandler::RemoveNamedArg (const std::string& Name)
{
   bool ReturnOK = true;
   CmdArg* TargetArg = NULL;
   for (CmdArg* Arg : I_CmdArgs) {
      if (Arg->MatchName(Name)) {
         TargetArg = Arg;
         break;
      }
   }
   if (TargetArg) {
      DelArg(TargetArg);
   } else {
      I_ErrorText = "Cannot delete argument '" + Name + "'. No such argument.";
      ReturnOK = false;
   }
   return ReturnOK;
}

void CmdHandler::AddArg (CmdArg* Arg)
{
   I_CmdArgs.push_back(Arg);
}

string CmdHandler::GetParameterFile(void)
{
   string Filename = "/tmp";
   struct passwd *PwdPtr = getpwuid (geteuid());
   if (PwdPtr) Filename += ("/" + string(PwdPtr->pw_name));
   if (access(Filename.c_str(),F_OK) != 0) mkdir (Filename.c_str(),0777);
   Filename += ("/" + I_Program + "_parameters");
   return Filename;
}

bool CmdHandler::SaveCurrent (void)
{
   bool ReturnOK = true;
   string FileName = GetParameterFile();
   FILE* ParamsFile = fopen (FileName.c_str(),"w");
   if (ParamsFile == NULL) {
      string Error = string(strerror(errno));
      I_ErrorText = "Unable to open parameter file '"
                                             + FileName + "': " + Error;
      ReturnOK = false;
   } else {
      for (CmdArg* Arg : I_CmdArgs) {
         if (!(Arg->DontSave())) {
            string ArgSpec = Arg->GetCurrentSpec();
            if (ArgSpec != "") {
               if (fprintf (ParamsFile,"%s\n",ArgSpec.c_str()) < 0) {
                  string Error = string(strerror(errno));
                  I_ErrorText = "Unable to write to parameter file '"
                                             + FileName + "': " + Error;
                  ReturnOK = false;
                  break;
               }
            }
         }
      }
      fclose (ParamsFile);
   }
   return ReturnOK;
}

bool CmdHandler::ReadPrevious (bool MustExist)
{
   bool ReturnOK = true;
   FILE* ParamsFile = NULL;
   string FileName = GetParameterFile();
   bool Exists = (access(FileName.c_str(), F_OK) != -1);
   if (Exists) {
      ParamsFile = fopen (FileName.c_str(),"r");
      if (ParamsFile == NULL) {
         string Error = string(strerror(errno));
         I_ErrorText = "Unable to open saved parameter file '"
                                             + FileName + "': " + Error;
         ReturnOK = false;
      }
   } else {
      if (MustExist) {
         ReturnOK = false;
         I_ErrorText = "Saved parameter file '" + FileName + "' does not exist";
      }
   }
   if (ParamsFile) {
      char Line[2048];
      while (fgets (Line,sizeof(Line),ParamsFile)) {
         Line[sizeof(Line) - 1] = '\0';
         int LastNonBlank = -1;
         for (unsigned int I = 0; I < sizeof(Line); I++) {
            if (Line[I] == '\0') break;
            if (Line[I] == '\n' || Line[I] == '\r') {
               Line[I] = '\0';
               break;
            }
            if (Line[I] != ' ') LastNonBlank = I;
         }
         Line[LastNonBlank + 1] = '\0';
         string LineString = Line;
         vector<string> Tokens;
         TcsUtil::Tokenize(LineString,Tokens);
         int Count = Tokens.size();
         if ((Count % 3) != 0) {
            I_ErrorText = "Incorrect token count for saved parameter values in "
                                                                  + FileName;
            ReturnOK = false;
            break;
         } else {
            int Index = 0;
            while (Index < Count) {
               string Name = Tokens[Index++];
               string Equals = Tokens[Index++];
               string Value = Tokens[Index++];
               if (Equals != "=") {
                  I_ErrorText = "Expected '=' got '" + Equals +
                            "' in saved parameter file " + FileName;
                  ReturnOK = false;
                  break;
               }
               bool Match = false;
               for (CmdArg* Arg : I_CmdArgs) {
                  if (Arg->MatchName(Name)) {
                     Match = true;
                     if (!Arg->SetPrevious(Value)) {
                        I_ErrorText = Arg->GetError();
                        ReturnOK = false;
                        break;
                     }
                  }
               }
               if (!Match) {
                  I_ErrorText = "Unexpected parameter name '" + Name +
                                "' in saved parameter file " + FileName;
                  ReturnOK = false;
               }
               if (!ReturnOK) break;
            }
         }
      }
      if (ferror(ParamsFile) && !feof(ParamsFile)) {
         string Error = string(strerror(errno));
         I_ErrorText = "Unable to read from saved parameter file '"
                                             + FileName + "': " + Error;
         ReturnOK = false;
      }
      fclose (ParamsFile);
   }
   
   return ReturnOK;
}

bool CmdHandler::IsInteractive (void)
{
   return I_Interactor->IsInteractive();
}

CmdInteractor* CmdHandler::GetInteractor (void)
{
   return I_Interactor;
}

bool CmdHandler::CheckSetup (void)
{
   bool ReturnOK = true;
   
   if (!I_Setup) {
   
      //  Check the consistency of the positional argument specification.
      //  This code runs through all the arguments. For each one that
      //  specifies a position (Posn), array element Flags[Posn - 1] is set
      //  to the address of that argument. This lets us spot duplicates
      //  and gaps in the set of positional arguments. (Note that the index
      //  into Flags is Posn - 1 because position numbers start at 1, a
      //  Posn value of 0 indicating a non-positional argument. (Annoyingly,
      //  I can't just declare Flags as a verying length array, because that
      //  C99 feature isn't in strict C++11.)
      
      int Nargs = I_CmdArgs.size();
      CmdArg** Flags = new CmdArg*[Nargs];
      for (int I = 0; I < Nargs; I++) {Flags[I] = NULL;}
      int MaxPosn = 0;
      for (CmdArg* Arg : I_CmdArgs) {
         int Posn = Arg->I_Posn;
         if (Posn < 0) {
            I_ErrorText = "Argument setup error. " + Arg->Description() +
                  " specifies a -ve position (" + std::to_string(Posn) + ")";
            ReturnOK = false;
            break;
         }
         if (Posn > Nargs) {
            I_ErrorText = "Argument setup error. " + Arg->Description() +
                                       " specifies too large a position (" +
                                                  std::to_string(Posn) + ")";
            ReturnOK = false;
            break;
         }
         if (Posn > 0) {
            int Index = Posn - 1;
            if (Flags[Index] != NULL) {
               I_ErrorText = "Argument setup error. " + Arg->Description() +
                  " and " + Flags[Index]->Description() +
                  " both specify position " + std::to_string(Posn);
               ReturnOK = false;
               break;
            }
            if (Posn > MaxPosn) MaxPosn = Posn;
            Flags[Index] = Arg;
         }
      }
      if (ReturnOK) {
         for (int I = 0; I < MaxPosn; I++) {
            if (Flags[I] == NULL) {
               I_ErrorText = "Argument setup error. Specified positions are "
                  "not sequential. No argument specifies position " +
                                                       std::to_string(I + 1);
               ReturnOK = false;
               break;
            }
         }
      }
      delete[] Flags;
   
   }
   if (ReturnOK) I_Setup = true;
   
   return ReturnOK;
}

void CmdHandler::RequestExit (void)
{
   I_ExitRequested = true;
}

bool CmdHandler::ExitRequested (void)
{
   return I_ExitRequested;
}

void CmdHandler::ListArgs (void)
{
   CmdInteractor* Interactor = GetInteractor();
   
   //  We pass through the arguments twice. First, a simple summary of all 
   //  the arguments. Then we run through them again, in the same order,
   //  giving more details.
   
   for (int Pass = 0; Pass < 2; Pass++) {
      
      //  Look at the positional parameters first, in order of position.
      
      if (Interactor) {
         Interactor->WriteLn("");
         if (Pass == 0) {
            Interactor->WriteLn("Argument summary:");
         } else {
            Interactor->WriteLn("In more detail:");
         }
         Interactor->WriteLn("");
      }
      string Summary = "";
      int NextPosn = 0;
      bool Found = true;
      while (Found) {
         NextPosn++;
         Found = false;
         for (CmdArg* Arg : I_CmdArgs) {
            if (Arg->I_Posn == NextPosn) {
               if (Pass == 0) {
                  Summary += Arg->I_Name + "(" + Arg->GetArgType() + ") ";
               } else {
                  string Text = "(" + Arg->GetArgType() + ") " + 
                                                       Arg->Description();
                  if (Interactor) Interactor->WriteLn(Text);
               }
               Found = true;
               break;
            }
         }
      }
      Summary += "| ";
      
      //  Now the rest (first, ignoring the 'hidden' arguments)
      
      for (CmdArg* Arg : I_CmdArgs) {
         if (Arg->I_Posn == 0) {
            if (!Arg->Hidden()) {
               if (Pass == 0) {
                  Summary += Arg->I_Name + "(" + Arg->GetArgType() + ") ";
               } else {
                  string Text = "(" + Arg->GetArgType() + ") " 
                                                         + Arg->Description();
                  if (Interactor) Interactor->WriteLn(Text);
               }
            }
         }
      }
      
      //  And finally, the 'hidden' arguments.
      
      Summary += "< ";
      for (CmdArg* Arg : I_CmdArgs) {
         if (Arg->I_Posn == 0) {
            if (Arg->Hidden()) {
               if (Pass == 0) {
                  Summary += Arg->I_Name + "(" + Arg->GetArgType() + ") ";
               } else {
                  string Text = "(" + Arg->GetArgType() + ") " 
                                                      + Arg->Description();
                  if (Interactor) Interactor->WriteLn(Text);
               }
            }
         }
      }
      Summary += ">";
      if (Pass == 0 && Interactor) SplitLine(Summary);
   }
}

void CmdHandler::SplitLine (const string& Text)
{
   CmdInteractor* Interactor = GetInteractor();
   if (Interactor) {
      int Width = Interactor->ScreenWidth();
      int Length = Text.size();
      int First = 0;
      if (Width <= 0) {
         //  If we don't know how the screen width, write out the whole string.
         Interactor->WriteLn(Text);
      } else {
         //  First is the index of the start of the remaining chars in Text.
         while (First < Length) {
            //  If we write a full screen line from what's left of the text,
            //  will we be breaking a word in two? Look at the next character.
            int TestPosn = First + Width - 1;
            if (TestPosn >= Length) {
               //  The remaining text will all fit on the screen.
               Interactor->WriteLn(Text.substr(First));
               break;  // All done.
            }
            if (Text[TestPosn] == ' ') {
               //  What comes next will be a space, so we're not breaking a
               //  word. Write the full screen width (don't care if there
               //  are spaces at the end). Then, skip to the next word.
               Interactor->WriteLn(Text.substr(First,Width));
               First = TestPosn + 1;
               while (Text[First] == ' ' && First < Length) First++;
            } else {
               //  If we write a full screen line, it will break a word.
               //  Work back until we find the start of the blanks and
               //  write up to there. If there are no blanks, we just have
               //  to break the word.
               int Ind = TestPosn - 1;
               while (Text[Ind] != ' ' && Ind > First) Ind--;
               if (Ind <= First) Ind = First + Width - 1;
               Interactor->WriteLn(Text.substr(First,Ind - First + 1));
               First = Ind + 1;
            }
         }
      }
   }
}

bool CmdHandler::ParseArgs (int Argc, char* Argv[])
{
   vector<string> Args;
   for (int I = 1; I < Argc; I++) {
      Args.push_back(string(Argv[I]));
   }
   return ParseArgs (Args);
}

bool CmdHandler::ParseArgs (const vector<string>& Args)
{
   bool ReturnOK = true;
   
   //  Make sure initial checks have been performed.
   
   ReturnOK = CheckSetup();
   
   int ArgCount = Args.size();
   int Index = 0;
   
   //  Initialise the arguments for this parse

   if (ReturnOK) {
      I_ExitRequested = false;
      for (CmdArg* Arg : I_CmdArgs) {
         if (!Arg->Clear()) {
            I_ErrorText = Arg->GetError();
            ReturnOK = false;
            break;
         }
       }
   }
   
   while (ReturnOK) {
      if (Index >= ArgCount) break;
      string Item = Args[Index];
      
      int EndIndex = Index;
      bool Equals = false;
      string Value = "";
      bool GotValue = false;
      string Name = Item;

      size_t EqPosn = Name.find('=');
      if (EqPosn != string::npos) {
         // Item includes an '='
         Equals = true;
         if (EqPosn + 1 < Name.size()) {
            //  Something follows '='. Must be the value
            Value = Name.substr(EqPosn + 1);
            GotValue = true;
         } else {
            //  Value must be the next item.
            if (Index + 1 < ArgCount) {
               //  Get value from next item
               EndIndex = Index + 1;
               Value = Args[EndIndex];
               GotValue = true;
            }
         }
         Name = Name.substr(0,EqPosn);
      } else {
         //  Item doesn't include '=', see if the next item is an '=' or
         //  starts with '='.
         if (Index + 1 < ArgCount) {
            Item = Args[Index + 1];
            if (Item.substr(0,1) == "=") {
               //  Starts with '='
               Equals = true;
               EndIndex = Index + 1;
               if (Item.size() > 1) {
                  //  And a value follows the '='
                  Value = Item.substr(1);
                  GotValue = true;
               } else {
                  //  Just a single '='. Value must be next arg
                  if (EndIndex + 1 < ArgCount) {
                     EndIndex++;
                     Value = Args[EndIndex];
                     GotValue = true;
                  }
               }
            }
         }
      }
      
      //  At this point, we've looked at the item string at Args[Index]
      //  If it included '=', or if the next string in the argument list
      //  started with '='. then Equals is set. If a value followed the '='.
      //  GotValue is set, and Value is that value. Name may turn out to be
      //  the name of a parameter, but really it's just Args[Index]
      //  including any leading '-' but shorn of any '=' and anything
      //  following. See if it actually might be a name, and if so of which
      //  argument.
      
      CmdArg* NameArg = NULL;
      for (CmdArg* Arg : I_CmdArgs) {
         if (Arg->MatchName(Name)) {
            NameArg = Arg;
            break;
         }
      }
      
      //  We can check a couple of things now. If there was an '=' found,
      //  then this must be a specification by name, and so the name must match
      //  that of one of the arguments. There must also be a value specified.
      
      if (Equals) {
         if (!GotValue) {
            ReturnOK = false;
            I_ErrorText = "No value follows '=' in given arguments";
            break;
         }
         if (NameArg == NULL) {
            I_ErrorText = "'" + Name +
               "' is set to a value, but is not a recognised argument name";
            ReturnOK = false;
            break;
         }
      }
      
      //  If an argument name matches, then this is an argument specified by
      //  name. Obviously! Now, what is its value? If there was a '=' then
      //  Equals will be set, and we'll already have a value string in Value,
      //  and GotValue will be true.
      
      if (NameArg) {
      
         //  Most arguments require a value. If this does, then if we don't
         //  have it already, it will be in the next item in the argument list.
         //  We see if there is such an item. And see if this named argument
         //  requires a value.
         
         bool ValueRequired = true;
         if (NameArg->ValueOptional()) ValueRequired = false;
         
         if (!GotValue) {
            if (Index + 1 < ArgCount) {
               Value = Args[Index + 1];
               GotValue = true;
            }
         }
         if (ValueRequired && !GotValue) {
            I_ErrorText = "No value specified for argument " +
                                               NameArg->Description();
            ReturnOK = false;
            break;
         }
         
         //  If this is an argument that requires a value, then it's easy,
         //  we just tell it to set the value we've got. Otherwise, we
         //  use the longer SetValue() call that lets it decide.

         bool UsedValue = !Equals;
         bool SetOK;
         if (ValueRequired) {
            SetOK = NameArg->SetValue(Name,Value);
         } else {
            SetOK = NameArg->SetValue(Name,Equals,Value,GotValue,&UsedValue);
         }
         if (!SetOK) {
            I_ErrorText = NameArg->GetError();
            ReturnOK = false;
            break;
         }
         if (!Equals && UsedValue) EndIndex = Index + 1;

      } else {
      
         //  If it's not a recognised argument name, then it must be a value
         //  specified positionally. So it should be the lowest numbered
         //  positional argument that hasn't been seen yet.
      
         Value = Name;
         CmdArg* PosnArg = NULL;
         int MinPosn = ArgCount + 1;
         for (CmdArg* Arg : I_CmdArgs) {
            if (!Arg->IsSet()) {
               if (Arg->I_Posn > 0 && Arg->I_Posn < MinPosn) {
                  MinPosn = Arg->I_Posn;
                  PosnArg = Arg;
               }
            }
         }
         if (PosnArg == NULL) {
            I_ErrorText = "'" + Value +
                "' does not correspond to any expected argument";
            ReturnOK = false;
         } else {
            Name = PosnArg->GetName();
            if (!PosnArg->SetValue(Name,Value)) {
               I_ErrorText = PosnArg->GetError();
               ReturnOK = false;
               break;
            }
         }
      }
      Index = EndIndex + 1;
   }
   
   //  Implement 'reset' by checking the internal 'Reset' arg.

   if (ReturnOK) {
      if (I_ResetArg) {
         if (I_ResetArg->Active()) {
            bool Reset = I_ResetArg->GetValue(&ReturnOK,&I_ErrorText);
            if (ReturnOK && Reset) {
               for (CmdArg* Arg : I_CmdArgs) {
                  Arg->Reset();
               }
            }
         }
      }
   }

   //  Similarly, implement 'prompt' by checking the internal 'Prompt' arg.

   if (ReturnOK) {
      if (I_PromptArg) {
         if (I_PromptArg->Active()) {
            if (I_PromptArg->IsSet()) {
               bool Prompt = I_PromptArg->GetValue(&ReturnOK,&I_ErrorText);
               if (ReturnOK) {
                  string OnOffDefault = Prompt ? "On" : "Off";
                  for (CmdArg* Arg : I_CmdArgs) {
                     if (!(Arg->Hidden())) Arg->SetPrompting(OnOffDefault);
                  }
               }
            }
         }
      }
   }

   //  Similarly, implement 'list' by checking the internal 'List' arg.
   
   if (ReturnOK) {
      if (I_ListArg) {
         if (I_ListArg->Active()) {
            bool List = I_ListArg->GetValue(&ReturnOK,&I_ErrorText);
            if (ReturnOK && List) {
               for (CmdArg* Arg : I_CmdArgs) {
                  if (!(Arg->Hidden())) Arg->SetListing(List);
               }
            }
         }
      }
   }
   
   //  Similarly, implement 'help' by checking the internal 'help' arg. But
   //  with 'help' it's probably confusing to carry on with prompting etc, so
   //  we treat 'help' as an exit request, as well as producing the requested
   //  help.
   
   if (ReturnOK) {
      if (I_HelpArg) {
         if (I_HelpArg->Active()) {
            bool Help = I_HelpArg->GetValue(&ReturnOK,&I_ErrorText);
            if (ReturnOK && Help) {
               ListArgs();
               RequestExit();
            }
         }
      }
   }

   return ReturnOK;
}

void CmdHandler::SetError (const string& ErrorText)
{
   I_ErrorText = ErrorText;
   I_ErrorReported = true;
}

string CmdHandler::GetError (void)
{
   return I_ErrorText;
}

#ifdef COMMAND_TEST

int main (int argc, char* argv[])
{

   CmdHandler TheHandler("CommandTest");
   
   int Posn = 1;
   FileArg GalaxyFileArg(TheHandler,"GalFile",Posn++,
      "Required,MustExist,NullOk",".fld","Name of file with galaxy positions");
   FileArg GuideFileArg(TheHandler,"GuideFile",Posn++,
      "Required,MustExist,NullOk",".fld","Name of file with star positions");
   FileArg OutputFileArg(TheHandler,"OutputFile",Posn++,"Required",".csv",
      "Name of output file");
   StringArg LabelArg(TheHandler,"Label",Posn++,"Required","",
      "Label to be written into output file");
   StringArg PlateIdArg(TheHandler,"PlateId",Posn++,"Required","",
      "Plate ID to be written into output file");
   StringArg DateTimeArg(TheHandler,"DateTime",Posn++,"Required","",
      "UT date and time, eg '2020 01 28 15 30 00.0'");

   RealArg RobotTempArg(TheHandler,"RobotTemp",Posn++,"",15.0,-10.0,60.0,
      "Temperature in deg C at which the plate is configured");
   RealArg ObsTempArg(TheHandler,"ObsTemp",Posn++,"",15.0,-10.0,60.0,
      "Temperature in deg C when the observation will be performed");
   
   FileArg DistortionFileArg(TheHandler,"2dFDistortion",0,"MustExist",
      "$TDF_DISTORTION,tdf_distortion1.sds",
      "Name of file giving 2dF distortion parameters");
   FileArg SkyFibreFileArg(TheHandler,"SkyFibres",0,"MustExist",
       "$SKY_FIBRES,SkyFibres.csv", "Name of file giving sky fibre positions");
   FileArg ProfitDirArg(TheHandler,"ProfitDir",0,"MustExist","$PROFIT_DIR",
      "Name of directory containing ProFit mask files");
   

   BoolArg TeleArg(TheHandler,"Tele",0,"",true,
      "Apply telecentricity corrections");
   BoolArg OffsetArg(TheHandler,"Mech",0,"",true,
      "Apply mechanical offset corrections");
   
   GalaxyFileArg.SetText("This contains the locations of the galaxies,\n"
                         "but can be left as a null.");
   TheHandler.ReadPrevious();
   if (!TheHandler.ParseArgs(argc,argv)) {
      printf ("Parse error: %s\n",TheHandler.GetError().c_str());
   }
   
   string GalaxyFile,GuideFile,OutputFile,Label,PlateId,DateTime;
   string DistortionFile,SkyFibreFile,ProfitDir;
   double RobotTemp,ObsTemp;
   bool Tele = false;
   bool Offset = false;
   
   bool Ok = true;
   string Error = "";

   GalaxyFile = GalaxyFileArg.GetValue(&Ok,&Error);
   GuideFile = GuideFileArg.GetValue(&Ok,&Error);
   OutputFile = OutputFileArg.GetValue(&Ok,&Error);
   
   Label = LabelArg.GetValue(&Ok,&Error);
   PlateId = PlateIdArg.GetValue(&Ok,&Error);
   DateTime = DateTimeArg.GetValue(&Ok,&Error);
   RobotTemp = RobotTempArg.GetValue(&Ok,&Error);
   ObsTemp = ObsTempArg.GetValue(&Ok,&Error);
   
   DistortionFile = DistortionFileArg.GetValue(&Ok,&Error);
   SkyFibreFile = SkyFibreFileArg.GetValue(&Ok,&Error);
   ProfitDir = ProfitDirArg.GetValue(&Ok,&Error);
   
   Tele = TeleArg.GetValue(&Ok,&Error);
   Offset = OffsetArg.GetValue(&Ok,&Error);
   if (Ok) {
      (void)TheHandler.SaveCurrent();
   } else {
      if (!TheHandler.ExitRequested()) {
         printf ("Error getting values: %s\n",TheHandler.GetError().c_str());
         printf ("Args report: %s\n",Error.c_str());
      }
   }
   
}

#endif

// -----------------------------------------------------------------------------

/*                        P r o g r a m m i n g  N o t e s

   o Now that this has been reworked to use strings throughout to hold values,
     which really simplifies the code, particularly the quite complex sequence
     used for DetermineValue(), which now replaces having that duplicated (not
     always accurately) in the GetValue() for each argument type, it really
     needs the comments fleshed out. It also needs some documentation.
 
   o The current parsing doesn't support some options used by UNIX utilities
     such as -DSYMBOL in c++ which defines the pre-processor variable SYMBOL.
     Nor would it support "-D SYMBOL=value" whith its three term syntax.
     Nor would it support the "-cvf filename" used by tar. All these could be
     added. Mostly, single letter named arguments need some special treatment.
 
   o I think it might be handy to be able to disable previous value reading and
     saving by passing the CommandHandler constructor a blank name.
 
   o I'm not sure at what point the error conditions should be reset - eg
     when should an argument clear its Error Text? If they're only used once,
     that's fine, but in principle an argument can be re-used; they can be
     used to prompt for simple values during a program - it doesn't have to
     be just command line handling.
 
   o Also on errors, I'm not completely happy with the interaction of error
     handling between the handler and the arguments. There are now two ways
     of finding out if there were errors getting the argument values - you
     can ask the overall Handler through AllOK(), but then have to explicitly
     check ExitRequested() to see if this not-OK is a real error or an exit
     request, or you can check the inherited OK status passed through all the
     GetValue() calls. Also there's a lot of copying of I_ErrorText values
     between the components.
 
   o In DetermineValue(), all calls to AllowedValue() are followed by calls
     to the helper CheckValidity() routine. This could be packaged more
     neatly, especially as the code checks each time for a helper.
 */
