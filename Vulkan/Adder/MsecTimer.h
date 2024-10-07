//
//                             M s e c  T i m e r . h
//
//  This provides a very basic timing facility. Essientially, you can create a
//  MsecTimer and then call its ElapsedMsec() method to get the time in msec
//  since it was created. You can also restart its timer by calling Restart().
//  And that's it.
//
//  This version uses gettimeofday() on UNIX-based systems, and used the
//  facilities provided by the GLFW window system on other systems, like
//  Windows. See Programming notes at the end.
//
//  22nd Feb 2024. First commented version using GLFW. KS.
//  23rd Feb 2024. Minor change - cast added - to placate VisualStudio compiler. KS.
//  14th Sep 2024. Now uses gettimeofday() for Unix-like systems to reduce need
//                 for GLFW on MacOS and Linux. KS.
//   4th Oct 2024. Replaced the GLFW code with Windows-specific code and a dummy
//                 for cases where we don't recognise the system. KS.

#ifndef __MsecTimer__
#define __MsecTimer__

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))

#include <sys/time.h>

class MsecTimer
{
public:
    MsecTimer() { Restart(); }
    ~MsecTimer() {}
    void Restart(void) {
        gettimeofday (&_startTime,NULL);
    }
    float ElapsedMsec(void) {
        struct timeval endTime;
        gettimeofday (&endTime,NULL);
        float Msec = (endTime.tv_sec - _startTime.tv_sec) * 1000.0 +
                    (endTime.tv_usec - _startTime.tv_usec) * 0.001;
        return Msec;
    }
private:
    struct timeval _startTime;
};

#elif defined (_WIN32)

//  This version is for Windows.

//  Annoyingly, windows.h defines max and min unless you tell it not to. Actually, it's
//  rather a shame to have a general-purpose include file like this that includes
//  Windows.h. Perhaps, at least on Windows, the code should be in a separate file,
//  and the FILETIME variable could be replaced by a double with the start msec time.

#define NOMINMAX
#include <windows.h>

class MsecTimer
{
public:
    MsecTimer() { Restart(); }
    ~MsecTimer() {}
    void Restart(void) {
        GetSystemTimePreciseAsFileTime(&_startTime);
    }
    float ElapsedMsec(void) {
        FILETIME currentTime;
        GetSystemTimePreciseAsFileTime(&currentTime);
        ULARGE_INTEGER start100nsec;
        ULARGE_INTEGER current100nsec;
        start100nsec.LowPart = _startTime.dwLowDateTime;
        start100nsec.HighPart = _startTime.dwHighDateTime;
        current100nsec.LowPart = currentTime.dwLowDateTime;
        current100nsec.HighPart = currentTime.dwHighDateTime;
        float Msec = float(current100nsec.QuadPart - start100nsec.QuadPart) / 10000.0f;
        return Msec;
    }
private:
    FILETIME _startTime;
};

#else

//  This is a dummy, only used if we can't recognise the system. It doesn't
//  work - it returns a negative elapsed time - but will will compile and link.

class MsecTimer
{
public:
    MsecTimer() {}
    ~MsecTimer() {}
    void Restart(void) {}
    float ElapsedMsec(void) { return -1.0; }
};


#endif

#endif

/*                       P r o g r a m m i n g   N o t e s
 
    o   This was originally written for MacOS and Linux, using gettimeofday().
        Then I discovered this didn't work on Windows and introduced a version
        using GLFW code, which was portable, but it was a nuisance to depend on
        GLFW and eventually I reworked it with specific Windows and *nix code.	
 
*/
