
//                      C o m m n a d  H a n d l e r . h
//
//  Function:
//    Provides a command line program with a way of handling arguments.
//
//  Description:
//    This defines a CommandHandler class that coordinates instances of a
//    set of classes derived from a base CmdArg class, instances of which
//    represent command line arguments of different types. For example, an
//    argument that represents a filename is handled by a FileArg instance,
//    an argument that represents a floating point number is handled by a
//    RealArg instance. The various arguments operate under the control of
//    a CommandHandler instance, which deals with parsing of the command line
//    parameters passed to the program. The individual argument instances all
//    provide a GetValue() call that can be used to return the value of the
//    corresponding argument.
//
//    The advantages these classes provide include:
//    o A very flexible command line syntax, allowing arguments to be specified
//      by name, and allowing boolean arguments to have negated forms, eg
//      either 'list' or 'nolist'.
//    o Command line constructs like "argument_name = value" are supported.
//    o Arguments can be specified by name or by position in the command line.
//    o Provided types include string,file,real, int and bool.
//    o Variants of these types can be easily implemented by inheriting from
//      these provided types.
//    o Default values for the arguments can be specified.
//    o Type-dependent syntax checking and clear error reporting.
//    o The ability to remember argument values from previous runs of the
//      program and use them as defaults.
//    o Support for controlling which values have to be specified on the
//      command line and which can be allowed to default.
//    o Prompting for argument values left off from the command line.
//    o Optional checking for the existence of files by FileArg instances.
//    o When prompting for file names, a FileArg supports filename completion
//      (using TAB and ^D characters).
//    o Files can be specified using environment variables.
//    o Numeric values can be checked against allowed limits.
//    o Numeric values can be specified as 'min' and 'max'.
//    o Help text can be provided for output when prompting for values.
//    o By default, 'help' on the command line outputs a summary of the options,
//      'prompt' forces prompting, 'list' lists all the values used, 'reset'
//      resets a parameter values to their defaults.
//
//    The result is that quite a lot of work can be packaged up into one
//    GetValue() call, and it's all done for you.
//
//  Author: Keith SHortridge, K&V (Keith@Knave&Varlet.com.au)
//
//  History:
//     22nd Jan 2021. Original version. KS.
//     31st Jan 2021. Revised internal structure to use strings internally
//                    to hold all values, moving almost all the code in
//                    the individual inheriting classes into the base CmdArg
//                    class. Added filename completion for filenames. KS.
//     24th Feb 2024. Revised protected and private setting for most instance
//                    variables, adding accessor routines in a number of
//                    cases. Substantial reorganisation and much expanded
//                    comments added. KS.
//     25th Jul 2024. Moved some RealArg and IntArg private routines to
//                    protected, and added GetRange(). Mostly to help with
//                    argument types derived from these. Corrected comments
//                    for FileArg about 'NullOK' option. KS.
//      9th Aug 2024. Added support for "!" (exit) and "?" (display more info)
//                    as responses to a prompt, and the new built-in 'help'
//                    argument. Some reworking of error handling so the handler
//                    knows about any errors reported by the individual
//                    arguments. KS.
//     16th Aug 2024. Introduced CmdArg::OutputHelpText(). KS.

#ifndef __CmdHandler__
#define __CmdHandler__

#include <string>
#include <list>
#include <vector>

//  Forward declarations

class CmdArg;
class BoolArg;
class CmdInteractor;

//  ----------------------------------------------------------------------------
//
//                           C m d  H a n d l e r
//
//  A CmdHandler (Command Handler) manages a set of arguments (instances of
//  classes like RealArg, StringArg, IntArg, BoolArg, FileArg, etc, all of
//  which inherit from the base CmdArg class), each of which usually corresponds
//  to one command line argument for a program. Usually, a program creates
//  a CmdHandler, then all the command arguments it will use, passing them a
//  reference to the Command handler in their constructors. Then the program
//  passes the command line arguments (argc,argv) to the command handler's
//  ParseArgs() routine, which works with the individual arguments to parse
//  the command line. Then, the GetValue() routine for each argument can be
//  called to get the value of that argument, prompting if necessary for any
//  missing argument value.
//
//  In addition, the Command Handler can save the argument values for reuse
//  the next time the program is run, and can retrieve them at the start of
//  the program.
//
//  The CommandHandler has three built-in boolean arguments, 'list', 'reset'
//  and 'prompt'. If a program doesn't want the functionality these offer,
//  or wants to implement its own arguments with those names, it can call
//  RemoveNamedArg() as required before creating its own arguments.

class CmdHandler {
public:
   //  Constructor. The string identifies the program when arguments are saved.
   CmdHandler (const std::string& Program);
   //  Destructor.
   ~CmdHandler();
   //  Read the previous argument values - call prior to parsing.
   bool ReadPrevious (bool MustExist = false);
   //  Parses the command line arguments passed to the program
   bool ParseArgs (int Argc, char* Argv[]);
   //  An alternative form of ParseArgs that uses a vector of strings
   bool ParseArgs (const std::vector<std::string>& Args);
   //  Save the current values of the arguments - use after calls to GetValue()
   bool SaveCurrent (void);
   //  Returns whether an exit has been requested.
   bool ExitRequested (void);
   //  Returns a description of the last error
   std::string GetError (void);
   //  Returns true if the program is interactive (ie if it can prompt)
   bool IsInteractive (void);
   //  Overrides the default interactor used for communicating with the user.
   void SetInteractor (CmdInteractor* Interactor);
   //  Deletes a named argument from those in use.
   bool RemoveNamedArg (const std::string& Name);
   //  Checks that everything is going well with both handler and arguments.
   bool AllOK (void);
protected:
   //  These routines are used by the individual arguments.
   friend class CmdArg;
   //  Delete an argument - called from an argument's destructor.
   void DelArg (CmdArg* Arg);
   //  Add an argument - called from an argument's constructor.
   void AddArg (CmdArg* Arg);
   //  Get access to the interactor used to communicate with the user.
   CmdInteractor* GetInteractor (void);
   //  Allows an argument to request an exit.
   void RequestExit(void);
   //  Records that an argument detected an error.
   void SetError (const std::string& Error);
private:
   //  Returns the name of the file used to save the argument values
   std::string GetParameterFile(void);
   //  Checks the setup for consistency prior to parsing arguments
   bool CheckSetup (void);
   //  Lists the arguments
   void ListArgs (void);
   //  Output a long line of text split to fit screen width.
   void SplitLine (const std::string& Text);
   //  True if the previous argument values have been read
   bool I_ReadPrevious;
   //  True if the comamnd handler has been setup and checked.
   bool I_Setup;
   //  Flags that an exit has been requested - a '!' response to a prompt.
   bool I_ExitRequested;
   //  Flags that an argument has requested an error.
   bool I_ErrorReported;
   //  The name of the program, as passed in the constructor.
   std::string I_Program;
   //  Description of the latest error.
   std::string I_ErrorText;
   //  The set of command arguments
   std::list<CmdArg*> I_CmdArgs;
   //  The built-in argument that handles 'reset'
   BoolArg* I_ResetArg;
   //  The built-in argument that handles 'prompt'
   BoolArg* I_PromptArg;
   //  The built-in argument that handles 'list'
   BoolArg* I_ListArg;
   //  The built-in argument that handles 'help'
   BoolArg* I_HelpArg;
   //  True if the interactor in use was supplied externally
   bool I_ExternalInteractor;
   //  The interactor used to communicate with the user.
   CmdInteractor* I_Interactor;
};

//  ----------------------------------------------------------------------------
//
//                           C m d  A r g  H e l p e r
//
//  A CmdArgHelper is the base class for specific user-supplied argument
//  helpers that can provide additional functionality for the existing arg
//  classes. Providing one of these for an argument like StringArg, for
//  example, may be simpler than creating a new class that inherits from
//  the argument class in question. A CmdArg can be introduced to a helper
//  using SetHelper(), and will then call the helper's CheckValidity() method
//  to provide additional checks on the supplied value for the argument. For
//  example, it could check a string against a set of allowed strings, or an
//  integer that needs to have an odd or an even value.

class CmdArgHelper {
public:
   //  Constructor;
    CmdArgHelper (void) {};
   //  Destructor;
   virtual ~CmdArgHelper() {};
   //  Checks validity of a supplied value. The value is passed as a string,
   //  and will have already passed the usual validity tests for the argument
   //  type - ie for IntArg it will represent a valid integer in the required
   //  range. If CheckValidity() rejects the value, it should set Reason to
   //  a description of the reason for doing so.
    virtual bool CheckValidity(const std::string& Value,std::string* Reason) {
        return true;
    };
    //  Returns additional help text to be used in response to '?'.
    virtual std::string HelpText(void) { return ""; }
};

//  ----------------------------------------------------------------------------
//
//                             C m d  A r g
//
//  The CmdArg class is the base class for the various argument types, and
//  does most of the work for them. The CmdArg class treats all values as
//  strings internally, and most of the code needed for the derived argument
//  types is merely connected with converting between these string values and
//  the actual type that they represent. The derived classes are also
//  responsible for validity checking for their argument types, which can
//  include interpreting special forms of the value, such as 'max' or 'min'
//  for numeric values or expanding environment variables used in file names
//  as done by FileArg. Each derived argument class is also expected to provide
//  a GetValue() call that returns the value of the argument in the expected
//  type (int, double, string, etc), and to provide a constructor that allows
//  reset values and limit values to be specified as examples of the type in
//  question.
//
//  A set of objects derived from CmdArg work together under the control of
//  a CommandHandler. Normally a Command Handler parses the command line
//  arguments passed to a program, working with its set of CmdArg objects,
//  and then the program can call the GetValue() method for each CmdArg
//  object to get the various argument values. If running interactively,
//  a CmdArg-based class object can prompt the user for the value of any
//  argument not explicitly specified on the command line, presenting a
//  sensible default value, or it can fall back on some other value that
//  it has been given - a default value, a previous value, or a 'reset'
//  value. Aguments can be specified on the command line by name or by
//  position.
//
//  (Note that it is assumed that there can always be a precise and reversible
//  conversion from the native type of any argument to a string and back. This
//  is obviously true for arguments whose values are basically strings, like
//  StringArg and FileArg, and is also true for IntArg, but it is not
//  self-evidently true for a RealArg. The trick is to support an 'exact'
//  string form for each argument, which is not necessarily the formatted
//  representation printf() or cout() would produce - it is more like a
//  serialisation of the internal representation. For most argument types the
//  'exact' string form is just that normal formatted form, eg '123' for the
//  integer 123, but it does not have to be, and the string values CmdArg works
//  with internally are always these 'exact' values. An exception is the call
//  DisplayValue(), which turns an internal 'exact' string into a formatted
//  value a user can understand, and which is null for most argument types,
//  but not for a RealArg.)
//
//  A CmdArg juggles a number of possible values for each argument:
//
//  o A 'Set' value specified in a set of arguments parsed by the Command
//    Handler. Usually these are the command line arguments for the program.
//  o A 'Default' value explicitly set in a SetDefault() call. Note that the
//    value of the reset argument is fixed once the CmdArg is created, but
//    the value for the default argument may change depending on circumstances.
//    A program may get the value of a file argument, open the file, and
//    depending on the file contents, the default for a later numeric
//    argument may change.
//  o A 'Previous' value, which is the value last used for this argument.
//    This may have been in a previous invocation of the program, since the
//    Command Handler can save and restore argument values.
//  o A 'Reset' value supplied in the argument constructor. This is the
//    value of last resort, used if nothing else has been specified.
//
//  Generally, these are used in the order of priority shown above. If a
//  value is specified explicitly on the command line, this value is used.
//  If no value was specified explicitly, then the CmdArg will try to determine
//  a default value. It will use a default value supplied explicitly by
//  the program, if there was one (on the assumption that the program knows
//  this is a sensible value), or if no explicit default was supplied, it will
//  use the previous value, if one is known, and as a last resort it will fall
//  back on the reset value specified in the constructor. If the program is
//  running interactively, the CmdArg can prompt the user for any value
//  unspecified on the command line, indicating in the prompt the default
//  value it has picked. The user can accept that, or supply a new value. If
//  the user supplies a value, that is the one that is used, so long as it
//  is valid.
//
//  Some control over this can be exercised:
//  o If an argument is flagged (in the Flags argument passed to the
//    constructor) as 'REQUIRED', then a value must be explicitly provided
//    by the user. If no valid value is provided (ie on the command line)
//    then a 'required' argument will always be prompted for, if possible.
//    If it cannot be prompted for, then the CmdArg will report an error.
//  o If a value is not specified on the command line, but a valid default
//    value is available (either an explicit default, the previous value or
//    the reset value) and the argument is not flagged as 'REQUIRED' the
//    CmdArg will usually use the default without prompting. The Command
//    Handler can force prompting for all unspecified arguments, usually
//    because 'prompt' was specified on the command line.
//  o A CommandHandler can force a 'reset' condition (usually because
//    'reset' was explicitly specified on the command line. In this case,
//    any previous value is forgotten, and in the absence of an explicitly
//    specified default, the CmdArg will fall back on the reset value for
//    any argument unspecified on the command line. (It will, of course,
//    prompt for the value if possible.)
//  o The Command Handler can turn off prompting for unspecified arguments,
//    usually because 'noprompt' was specified on the command line. The
//    user can make use of this if they know that all the defaults, even
//    for required arguments, are valid. (Usually the effect is to use the
//    previous values for all unspecified arguments.)

class CmdArg {
public:
   //  -------------------------------------------------------------------------
   //  These are the routines that can be called from the program that creates
   //  the command handler and the arguments. These are the routines common to
   //  all the arguments. Note that these do not include the GetValue() routine
   //  each argument is expected to provide, as that returns the value in the
   //  specific type for the argument. The derived classes provide that, and
   //  are expected to use the protected DetermineValue() call to do most of
   //  the work. DetermineValue() provides a string version of the value,
   //  and the derived classes turn that into their native tyep. Similarly,
   //  these routines do not include the SetDefault() call that can provide a
   //  native default value. Nothing does by CmdArg uses native argument types
   //  like int, bool, real - it works entirely in strings. Usually, all a using
   //  program has to do is contruct the arguments it needs, get the command
   //  handler to parse the command arguments, and then call GetValue() for
   //  each argument. If it works out a sensible default for the argument, it
   //  can call SetDefault() before calling GetValue().
   //  -------------------------------------------------------------------------
   //  Constructor.
   //  Name    is name of argument
   //  Posn    is position (0 => not positional, positions start at 1)
   //  Flags   flags - a string of comma/space separated options
   //  Prompt  prompt used if value has to be prompted for
   //  Text    more detailed explanatory text (currently unused)
   CmdArg(CmdHandler& Handler,const std::string& Name,int Posn,
          const std::string& Flags, const std::string& Prompt,
          const std::string& Text);
   //  Destructor.
   virtual ~CmdArg();
   //  Can be called to modify the text associated with an argument
   void SetText (const std::string& Text);
   //  Can be called to modify the prompt associated with an argument
   void SetPrompt (const std::string& Prompt);
   //  Can be called to modify the position associated with an argument
   void SetPosn (int Posn);
   //  Supplies a user-written helper for the argument.
   void SetHelper (CmdArgHelper* Helper);
   //  Returns the description of the latest error.
   std::string GetError(void);
   //  Returns the name of the argument.
   std::string GetName (void) { return I_Name; }
   //  Returns a description of the argument.
   std::string Description (void);
protected:
   //  -------------------------------------------------------------------------
   //  These are routines called by the base CmdArg() routines that may need
   //  to be overriden by the derived argument routines. These cover things
   //  such as validity checks and checks on ranges and for specific strings
   //  (eg 'max' and 'min' for numeric arguments). They also cover routines
   //  that handle cases where - as for a real argument - a conventional
   //  formatted string may not match the internal representation perfectly.
   //  For those latter routines, an 'exact' string value is a string that
   //  can be converted back to the internal binary representation of the number
   //  precisely at the bit level, which a 'display' string is one that
   //  makes sense to a user, but may not be precisely interchangeable with
   //  the internal binary representation. For example, a real argument might
   //  use a hex string that matches the bytes of its internal binary value
   //  as its 'exact' value string, and a conventionally formatted string (as
   //  produced by cout(), or printf() with %f or %g formats) as its 'display'
   //  value string.
   //  -------------------------------------------------------------------------
   //  Given a string value (exact or display) returns a display value.
   virtual std::string DisplayValue (const std::string& Value);
   //  Returns the default value for an argument whose value was not specified.
   //  Only applies to arguments where a value need not be specified, such
   //  as boolean arguments, where this routine will return "true".
   virtual std::string UnspecifiedValue(void);
   //  Passed a value, returns the negated version of it. Only needed for
   //  arguments (such as bools) that support negation, which would return
   //  'false' if passed 'true' and vice-versa.
   virtual std::string NegateValue(const std::string& Value);
   //  Returns true if the value string passed (exact or display) has a
   //  valid syntax. This does not check if that value is allowed (ie it
   //  would return true for a real value of "-1.0" even if the allowed range
   //  were from 0.0 to 10.0).
   virtual bool ValidValue (const std::string& Value);
   //  Returns a string describing the requirements for an acceptable value
   //  for the argument.
   virtual std::string Requirement (void);
   //  Allows an argument to modify a string so that special cases such as
   //  'max' and 'min' for numeric arguments can be supported. An IntArg
   //  with a range of 0 to 10 passed 'max' would return '10'. The string
   //  returned should be an 'exact' value string.
   virtual void ExpandValue (std::string* Value);
   //  Returns true if the string passed is an allowed value for the argument,
   //  being both valid and passing all other requirements (eg, a FileArg
   //  would test to see if a named file had to exist or not, and if it did.)
   virtual bool AllowedValue (const std::string& Value);
   //  -------------------------------------------------------------------------
   //  These are utility routines provided for the use of the Command Handler
   //  and the derived argument classes. These should not need to be overriden
   //  by derived classes. Note that many of these are essentially accessor
   //  routines that wrap up access to the various private instance variables.
   //  There are no protected instance variables - that keeps things tidier.
   //  -------------------------------------------------------------------------
   //  Tells the argument that the command handler is closing down
   void CloseHandler (CmdHandler* Handler);
   //  Clears the argument ready for a new parsing or command line values
   bool Clear (void);
   //  Tells the argument to ignore any existing values.
   void Reset (void);
   //  Returns a "name = value" string giving the value of the argument.
   std::string GetCurrentSpec (void);
   //  Sets a string version of the default value - used by SetDefault()
   void SetDefaultValue (const std::string& Default);
   //  Determine the current value for the argument - used by GetValue()
   bool DetermineValue (std::string* CurrentValue);
   //  Output any available help text.
   void OutputHelpText(void);
   //  Returns true if the argument can be specified by its name only.
   bool ValueOptional (void) { return I_Valopt; }
   //  Returns true if the argument has to be specified explicitly by the user.
   bool Required (void) { return I_Flags & REQUIRED; }
   //  Returns true if the argument has not been 'deleted' by the CmdHandler.
   bool Active (void) { return I_IsActive; }
   //  Returns true if the current value (I_Value) has been set.
   bool IsSet (void) { return I_IsSet; }
   //  True if the argument is 'hidden' - it is not prompted for or listed
   bool Hidden (void) { return I_Flags & HIDDEN; }
   //  True if this is an internal argument like 'list' used by the CmdHandler.
   bool Internal (void) { return I_Flags & INTERNAL; }
   //  True if the argument can be specified in a negated form (eg 'nolist')
   bool Negatable (void) { return I_Flags & NEGATABLE; }
   //  True if the argument is required to have a value specified
   bool ValReq (void) { return I_Flags & VALREQ; }
   //  True if the value for the argument is optional
   bool ValOpt (void) { return I_Flags & VALOPT; }
   //  Set the argument as being optional or not.
   void SetValOpt (bool Valopt) { I_Valopt = Valopt; }
   //  Set the 'reset' vaue for the argument.
   void SetReset (const std::string& Reset) { I_Reset = Reset; }
   //  Set the argument to list its value for the user when GetValue is called.
   void SetListing (bool Listing = true);
   //  Returns true if the argument has been set to list its value.
   bool IsListing (void) { return I_Listing; }
   //  Set the prompting for the argument - 'on', 'off' or 'default'
   void SetPrompting (const std::string& OnOffDefault);
   //  True if the argument is allowed to prompt (depends on the CmdHandler)
   bool CanPrompt (void);
   //  True if the argument value is not to be saved.
   bool DontSave (void) { return I_Flags & NOSAVE; }
   //  True if the argument is a file name (this affects the prompting)
   bool IsFile (void) { return I_Flags & IS_FILE; }
   //  Sets a specific bit in I_Flags.
   void SetFlag (unsigned long Flag) { I_Flags |= Flag; }
   //  Set the string describing any error in the constructor.
   void SetConstructError (const std::string& Text) { I_ConstructError = Text;}
   //  Set the string describing the latest error.
   void SetError (const std::string& Text);
   //  Get the list of Flag options not recognised by the base class.
   std::vector<std::string> GetUnknownFlags (void);
   //  Set the list of Flag options not recognised by the base class.
   void SetUnknownFlags (const std::vector<std::string>& Flags);
   //  Set the 'previous' value for the argument.
   bool SetPrevious (const std::string& Value);
   //  Output a message to the user.
   bool TellUser (const std::string& ReportString);
   //  Prompt the user and return their reply.
   std::string PromptUser (const std::string& Default);
   //  Modifies a string value so it can be recorded (quotes strings if needed)
   std::string FormatForRecord (const std::string& Value);
   //  Returns the argument type ('file','string', etc)
   std::string GetArgType (void);
   //  Sets the argument type ('file','string', etc)
   void SetArgType (const std::string& Type);
   //  Returns true if a name matches that used by the argument
   bool MatchName (const std::string& Name);
   //  Sets the value where a value and an argument name were both given.
   bool SetValue (const std::string& Name, const std::string& Value);
   //  Sets the value in cases where values are optional.
   bool SetValue (const std::string& Name, bool Equals,
                    const std::string Value, bool GotValue, bool* UsedValue);
   //  Returns the negated form of the argument's name.
   bool NegName (const std::string& Name);
   //  Returns the non-negated (ie the usual) form of the argument name.
   bool PosName (const std::string& Name);
   //  The Flag values coresponding to the strings that can be passed
   //  to the constructor in the Flags argument. These are only used internally
   //  (and a derived class can use a SetFlag() call to set one of these).
   static const unsigned long REQUIRED = 0x1;    // Must be specified
   static const unsigned long VALOPT = 0x2;      // Does not need a value
   static const unsigned long VALREQ = 0x4;      // Must have a value a value
   static const unsigned long HIDDEN = 0x8;      // Not normally visible to user
   static const unsigned long NOSAVE = 0x10;     // Value should not be saved
   static const unsigned long NEGATABLE = 0x20;  // Argument name can be negated
   static const unsigned long IS_FILE = 0x40;    // Value is a file name
   static const unsigned long INTERNAL = 0x80;   // Set only by command handler
   //  The CmdHandler has access to a number of these protected routines.
   friend class CmdHandler;
private:
   //  Pointer to the CmdHandler this argument is working with.
   CmdHandler* I_Handler;
   //  Name of the Argument.
   std::string I_Name;
   //  Prompt associated with the argument.
   std::string I_Prompt;
   //  More detailed text associated with the argument (not yet supported).
   std::string I_Text;
   //  Argument type (so it can be included in a list)
   std::string I_ArgType;
   //  Position (from 1 up) for a positional argument. 0 => non-positional.
   long I_Posn;
   //  Flags set for the argument.
   unsigned long I_Flags;
   //  Address of any user-supplied helper.
   CmdArgHelper* I_Helper;
   //  True if a previous value has been set - the value used last time.
   bool I_PreviousSet;
   //  String version of the previous value.
   std::string I_Previous;
   //  True if a value has been set by the command handler - from a command line
   bool I_IsSet;
   //  Value set by the command handler, as a string
   std::string I_SetValue;
   //  True if a default value has been set dynamically.
   bool I_DefaultSet;
   //  Default value, as a string - a default that can be set dynamically
   std::string I_Default;
   //  Reset value, as a string - a default set in the constructor.
   std::string I_Reset;
   //  The value returned by GetValue() - ie the current value.
   std::string I_Value;
   //  True if the argument is active - ie has not been explicitly disabled.
   bool I_IsActive;
   //  True if the argument does not necessarily require a value.
   bool I_Valopt;
   //  True if the argument is to report the value it supplies in GetValue().
   bool I_Listing;
   //  True if prompting is on, even for nominally hidden arguments.
   bool I_PromptOn;
   //  True if prompting is off for all arguments.
   bool I_PromptOff;
   //  Describes any error that occurred during execution of the constructor.
   std::string I_ConstructError;
   //  Description of the last error associated with the argument.
   std::string I_ErrorText;
   //  Contains all the unrecognised items passed to the constructor in Flags.
   std::vector<std::string> I_UnknownFlags;
};

//  ----------------------------------------------------------------------------
//
//                          S t r i n g  A r g
//
//  A StringArg inherits from CmdArg, and implements a straightford argument
//  whose value is just a string. This means that the basic CmdArg does all
//  that's required for this, and all a StringArg has to do is provide a
//  constructor with a string 'reset' value, and GetValue() and SetDefault()
//  calls that work with strings.

class StringArg : public CmdArg {
public:
   StringArg (CmdHandler& Handler,const std::string& Name,int Posn = 0,
      const std::string& Flags = "",const std::string& Reset = "",
      const std::string& Prompt = "",const std::string& Text = "");
   virtual ~StringArg();
   virtual std::string GetValue (bool* OK,std::string* Error);
   virtual void SetDefault (const std::string& Default);
};

//  ----------------------------------------------------------------------------
//
//                          F i l e  A r g
//
//  A FileArg inherits from StringArg, and implements an argument whose
//  value is a string, but a string that will be used as the name of a file.
//  A FileArg accepts a couple of additional strings in the Flags argument:
//  'NullOK' indicates that a null value (a string with zero length, ie "")
//  is acceptable - the program can handle this as it chooses - and 'MustExist'
//  indicates that if the value supplied is not null, it must be the name of
//  a file that already exists.
//
//  A FileArg has a constructor that accepts a 'reset' value that is a
//  comma- or space-separated set of strings. The constructor uses the first
//  of these that is an acceptable option. Each can include the name of an
//  environment variable, preceded by a '$', and if this is the case, the
//  constructor expands the environment variable. (If the environment variable
//  is not defined, the this string is not an acceptable option.) If 'MustExist'
//  was specified in Flags, and the string is not the name of an existing
//  string, then this string is not an acceptable option. Eventually, either
//  the constructor will have an acceptable reset value, or it will not, and
//  usually the argument will end up prompting for the value if it is not
//  specified in some other way.
//
//  When a FileArg prompts the user for a file name, it tells the interactor
//  used to communicate with the user that this is a file name, and the
//  interactor can use this to do things like filename completion.
//
//  The FileArg uses the GetValue() and SetDefault() routines provided by
//  StringArg, but provides its own AllowedValue(), ExpandValue() and
//  Requirement() routines.

class FileArg : public StringArg {
public:
   FileArg (CmdHandler& Handler,const std::string& Name,int Posn = 0,
      const std::string& Flags = "",const std::string& Reset = "",
      const std::string& Prompt = "",const std::string& Text = "");
   virtual ~FileArg();
protected:
   virtual bool AllowedValue (const std::string& Value);
   virtual void ExpandValue (std::string* Value);
   virtual std::string Requirement (void);
private:
   bool NullOk (void) { return I_FileFlags & NULL_OK; }
   bool MustExist (void) { return I_FileFlags & MUST_EXIST; }
   bool FileExists (const std::string& FileName);
   static const unsigned long MUST_EXIST = 0x10; // File must exist already
   static const unsigned long NULL_OK = 0x100;   // Filename can be null
   unsigned long I_FileFlags;
   void ExpandTildePath (std::string* Path);
   std::vector<std::string> I_ResetValues;
};

//  ----------------------------------------------------------------------------
//
//                          B o o l  A r g
//
//  A BoolArg inherits from CmdArg, and implements an argument whose
//  value is a boolean value, accepting "true","false","yes" or "no" as
//  explicit values. A BoolArg does not have to be given an associated value
//  on the command line, and is normally negatable. That is, a BoolArg such as
//  the 'prompt' argument built-in to the command handler can be specified on
//  the command line in any of the following ways:
//
//  prompt
//  noprompt
//  prompt = yes
//  prompt = no
//  prompt true
//  prompt false
//
//  If you really want, you can even have:
//
//  noprompt = false
//
//  which is the same as just 'prompt'.
//
//  A BoolArg provides its own GetValue() and SetDefault() and overrides the
//  ValidValue(), AllowedValue(), NegateValue(), UnspecifiedValue() and
//  Requirement() calls provided byCmdArg.

class BoolArg : public CmdArg {
public:
   BoolArg (CmdHandler& Handler,const std::string& Name,int Posn = 0,
      const std::string& Flags = "",bool Reset = false,
      const std::string& Prompt = "",const std::string& Text = "");
   virtual ~BoolArg();
   virtual bool GetValue (bool* OK,std::string* Error);
   void SetDefault (bool Default);
protected:
   virtual bool ValidValue (const std::string& Value);
   virtual bool AllowedValue (const std::string& Value);
   virtual std::string NegateValue(const std::string& Value);
   virtual std::string UnspecifiedValue(void);
   virtual std::string Requirement (void);
private:
   bool CheckValidValue (const std::string& Value, bool* BoolValue);
};

//  ----------------------------------------------------------------------------
//
//                          I n t  A r g
//
//  An IntArg inherits from CmdArg, and implements an argument whose value is
//  an integer. An IntArg has a constructor that includes three integer values:
//  a 'reset' value, and a maximum and minimum value for the argument. It
//  provides its own GetValue() and SetDefault() routines, which work with
//  integers, and adds a SetRange() routine that can be used to override the
//  range values set in the constructor. If the Max and Min values are the
//  same, no range checking is performed. It overrides the ValidValue(),
//  AllowedValue(), ExpandValue() and Requirement() routines from CmdArg to
//  provide routines that understand that the strings used for the values
//  need to represent formatted integers. An integer can be formatted with
//  full precision, so the 'exact' and 'display' values are just conventionally
//  formatted strings.

class IntArg : public CmdArg {
public:
   IntArg (CmdHandler& Handler,const std::string& Name,int Posn = 0,
      const std::string& Flags = "",long Reset = 0,long Min = 0, long Max = 0,
      const std::string& Prompt = "",const std::string& Text = "");
   virtual ~IntArg();
   virtual long GetValue (bool* OK,std::string* Error);
   void SetDefault (long Default);
   virtual void SetRange (long Min, long Max);
protected:
   virtual bool ValidValue (const std::string& Value);
   virtual bool AllowedValue (const std::string& Value);
   virtual void ExpandValue (std::string* Value);
   virtual std::string Requirement (void);
   virtual bool CheckValidValue (const std::string& Value, long* IntValue);
   virtual std::string FormatInt (long Value);
   virtual void GetRange (long* Min, long* Max);
private:
   long I_Min;
   long I_Max;
};

//  ----------------------------------------------------------------------------
//
//                          R e a l  A r g
//
//  An IntArg inherits from CmdArg, and implements an argument whose value is
//  a floating point number, represented internally as a double. A RealArg has
//  a constructor that includes three double values: a 'reset' value, and a
//  maximum and minimum value for the argument. It provides its own GetValue()
//  and SetDefault() routines, which work with doubles, and adds a SetRange()
//  routine that can be used to override the range values set in the
//  constructor. If the Max and Min values are the same, no range checking is
//  performed. It overrides the ValidValue(), AllowedValue(), ExpandValue(),
//  Requirement() and DisplayValue() routines from CmdArg to provide routines
//  that understand that the strings used for the values need to represent
//  formatted floating point numbers. A floating point number cannot in general
//  be formatted without loss of precision, so while the 'display' values
//  are just conventionally formatted strings, the 'exact' values have to be
//  strings that represent the internal binary double value as a set of hex
//  digits.

class RealArg : public CmdArg {
public:
   RealArg (CmdHandler& Handler,const std::string& Name,int Posn = 0,
      const std::string& Flags = "",double Reset = 0.0,double Min = 0.0,
      double Max = 0.0, const std::string& Prompt = "",
      const std::string& Text = "");
   virtual ~RealArg();
   virtual double GetValue (bool* OK,std::string* Error);
   void SetDefault (double Default);
   virtual void SetRange (double Min, double Max);
protected:
   virtual bool ValidValue (const std::string& Value);
   virtual bool AllowedValue (const std::string& Value);
   virtual void ExpandValue (std::string* Value);
   virtual std::string Requirement (void);
   virtual std::string DisplayValue (const std::string& Value);
   virtual bool CheckValidValue (const std::string& Value, double* RealValue);
   virtual std::string FormatReal (double Value);
   virtual void GetRange (double* Min,double *Max);
private:
   std::string FormatRealExact (double Value);
   bool ReadRealExact (const std::string& Value,double* RealValue);
   bool HexCharChar2Char (char Ch1,char Ch2,unsigned char* Ch);
   double I_Min;
   double I_Max;
};

//  ----------------------------------------------------------------------------
//
//                      C m d  I n t e r a c t o r
//
//  A command handler needs to interact with the user, to output messages, to
//  prompt, to read responses, etc. Because different programs will run in
//  different environments, it uses a CmdInteractor class to do this. The
//  standard code provides a CmdInteractor that assumes it is being run in
//  a standard manner for a command line program, from a terminal application.
//  Programs where this isn't the case can provide their own class that
//  inherits from CmdInteractor and use the handler's SetInteractor() method
//  to override the default interactor. An interactor can operate in interactive
//  mode, with a user that it can prompt for input, or in a batch mode where
//  this is not possible. The default version checks this for itself, a
//  different implementation may need to be called through SetInteractive().

class CmdInteractor {
public:
   //  Constructor
   CmdInteractor (void);
   //  Destructor
   virtual ~CmdInteractor();
   //  Explicitly tell the interactor if it can interact with a user or not.
   virtual void SetInteractive (bool CanInteract = true);
   //  Returns true if operating in interactive mode.
   virtual bool IsInteractive (void);
   //  Writes the supplied text.
   virtual bool Write (const std::string& Text);
   //  Writes the supplied text followed by a new line character.
   virtual bool WriteLn (const std::string& Text = "");
   //  Reads a line from the user, with an optional prompt first. If IsFile
   //  is set, the response is expected to be a filename, allowing the use
   //  of filename completion.
   virtual bool ReadLn (std::string& Text,
                     const std::string& Prompt = "",bool IsFile = false);
   //  Get the width of the screen in characters. Returns zero if unknown.
   virtual int ScreenWidth (void);
   //  Returns a description of the most recent error.
   std::string GetError (void);
protected:
   int GetScreenWidth (void);
private:
   bool I_IsInteractive;
   std::string I_ErrorText;
};


#endif
