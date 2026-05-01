// Platform-specific font loading for Windows using GDI32.
//go:build windows

package loader

import (
	"fmt"
	"syscall"
	"unsafe"
)

var (
	gdi32                   = syscall.NewLazyDLL("gdi32.dll")
	procAddFontResourceW    = gdi32.NewProc("AddFontResourceW")
	procRemoveFontResourceW = gdi32.NewProc("RemoveFontResourceW")
	user32                  = syscall.NewLazyDLL("user32.dll")
	procSendMessageW        = user32.NewProc("SendMessageW")
	procSendMessageTimeoutW = user32.NewProc("SendMessageTimeoutW")
)

const (
	hwndBroadcast   = uintptr(0xFFFF)
	wmFontChange    = 0x001D
	smtoAbortIfHung = 0x0002
)

func loadWindows(srcPath string) (string, error) {
	ptr, err := syscall.UTF16PtrFromString(srcPath)
	if err != nil {
		return "", err
	}
	r, _, _ := procAddFontResourceW.Call(uintptr(unsafe.Pointer(ptr)))
	if r == 0 {
		return "", fmt.Errorf("AddFontResourceW failed for %s", srcPath)
	}
	// Notify all windows that fonts changed.
	procSendMessageW.Call(hwndBroadcast, wmFontChange, 0, 0) //nolint:errcheck
	return srcPath, nil
}

func unloadWindows(path string) error {
	ptr, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return err
	}
	r, _, _ := procRemoveFontResourceW.Call(uintptr(unsafe.Pointer(ptr)))
	if r == 0 {
		return fmt.Errorf("RemoveFontResourceW failed for %s", path)
	}
	procSendMessageW.Call(hwndBroadcast, wmFontChange, 0, 0) //nolint:errcheck
	return nil
}

func runFcCache() {
	// fc-cache is a Linux tool; no-op on Windows.
}

// cacheRefreshWindows broadcasts WM_FONTCHANGE to all windows so they reload fonts.
func cacheRefreshWindows() {
	procSendMessageTimeoutW.Call(
		hwndBroadcast,
		wmFontChange,
		0,
		0,
		smtoAbortIfHung,
		1000,
		0,
	) //nolint:errcheck
}
