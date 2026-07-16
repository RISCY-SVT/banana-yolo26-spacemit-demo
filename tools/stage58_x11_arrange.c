#include <X11/Xlib.h>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned long parse_unsigned(const char* text, const char* name) {
    char* end = NULL;
    errno = 0;
    unsigned long value = strtoul(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0') {
        fprintf(stderr, "invalid %s: %s\n", name, text);
        exit(2);
    }
    return value;
}

int main(int argc, char** argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s WINDOW_ID X Y WIDTH HEIGHT\n", argv[0]);
        return 2;
    }
    Display* display = XOpenDisplay(NULL);
    if (display == NULL) {
        fprintf(stderr, "cannot open X display\n");
        return 1;
    }
    const Window window = (Window)parse_unsigned(argv[1], "window id");
    const int x = (int)parse_unsigned(argv[2], "x");
    const int y = (int)parse_unsigned(argv[3], "y");
    const unsigned int width = (unsigned int)parse_unsigned(argv[4], "width");
    const unsigned int height = (unsigned int)parse_unsigned(argv[5], "height");
    XMoveResizeWindow(display, window, x, y, width, height);
    XRaiseWindow(display, window);
    XFlush(display);
    XCloseDisplay(display);
    return 0;
}
