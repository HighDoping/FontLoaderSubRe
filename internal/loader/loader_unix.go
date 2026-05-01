// Platform stubs for non-Windows builds.
//go:build !windows

package loader

import "os/exec"

func loadWindows(_ string) (string, error) { panic("not windows") }
func unloadWindows(_ string) error         { panic("not windows") }

func runFcCache() {
	exec.Command("fc-cache", "-f").Run() //nolint:errcheck
}

// cacheRefreshWindows is a no-op on non-Windows platforms.
func cacheRefreshWindows() {}
