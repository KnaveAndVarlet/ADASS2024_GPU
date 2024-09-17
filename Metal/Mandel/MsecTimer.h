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

#else

#include <GLFW/glfw3.h>

class MsecTimer
{
public:
    MsecTimer() { Restart(); }
    ~MsecTimer() {}
    void Restart(void) {
        _startTime = glfwGetTime();
    }
    float ElapsedMsec(void) {
        float Msec = float((glfwGetTime() - _startTime) * 1000.0);
        return Msec;
    }
private:
    double _startTime;
};

#endif

#endif

/*                       P r o g r a m m i n g   N o t e s
 
    o   This was originally written for MacOS and Linux, using gettimeofday().
        Then I discovered this didn't work on Windows and introduced the GLFW
        code, as the Mac and Linux systems I used had GLFW, as did my Windows
        system, so this worked on everything. But then I realised it was
        going to be a nuisance to depend on GLFW and added the #ifdef tests
        for Mac/Linux. I suppose I should really remove GLFW from the Windows
        code and replace it with Windows-specific code. One day..
 
*/
