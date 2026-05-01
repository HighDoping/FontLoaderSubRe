.PHONY: all build-unix build-cli build-win clean

BINARY_GUI    := FontLoaderSubRe
BINARY_CLI    := Indexer
CMD_GUI       := ./cmd/fontloader
CMD_CLI       := ./cmd/cli_indexer
FYNE          := $(shell go env GOPATH)/bin/fyne
RSRC          := $(shell go env GOPATH)/bin/rsrc
ICON          := resources/icon.ico
SYSO          := $(CMD_GUI)/resource.syso

# ── macOS ─────────────────────────────────────────────────────────────────────
build-unix: clean $(BINARY_GUI) $(BINARY_CLI)

$(BINARY_GUI):
	go build -o $(BINARY_GUI) $(CMD_GUI)

$(BINARY_CLI):
	go build -o $(BINARY_CLI) $(CMD_CLI)

# Optional: package as .app bundle using fyne tool
bundle-mac:
	cd $(CMD_GUI) && $(FYNE) package --os darwin --release

# ── CLI only ──────────────────────────────────────────────────────────────────
build-cli:
	go build -o $(BINARY_CLI) $(CMD_CLI)

# ── Windows  ──────────────────────────────────────────────────────────────────
build-win: $(SYSO)
	go build -ldflags="-H windowsgui" -o $(BINARY_GUI).exe $(CMD_GUI)
	go build -o $(BINARY_CLI).exe $(CMD_CLI)
#rm -f $(SYSO)

$(SYSO): $(ICON)
	$(RSRC) -ico $(ICON) -o $(SYSO)
bundle-win:
	cd $(CMD_GUI) && $(FYNE) package --os windows --release
# ── All ───────────────────────────────────────────────────────────────────────
all: build-unix build-cli

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -f $(BINARY_GUI) $(BINARY_GUI).exe
	rm -f $(BINARY_CLI) $(BINARY_CLI).exe
	rm -f $(SYSO)
	rm -rf FontLoaderSubRe.app
