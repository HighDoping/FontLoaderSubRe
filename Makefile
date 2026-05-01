.PHONY: all build-unix build-cli build-win clean

BINARY_GUI    := FontLoaderSubRe
BINARY_CLI    := cli_indexer
CMD_GUI       := ./cmd/fontloader
CMD_CLI       := ./cmd/cli_indexer
FYNE          := $(shell go env GOPATH)/bin/fyne

# ── macOS ──────────────────────────────────────────────────────────────────────
build-unix: clean $(BINARY_GUI) $(BINARY_CLI)

$(BINARY_GUI):
	go build -o $(BINARY_GUI) $(CMD_GUI)

$(BINARY_CLI):
	go build -o $(BINARY_CLI) $(CMD_CLI)

# Optional: package as .app bundle using fyne tool
bundle-mac:
	cd $(CMD_GUI) && $(FYNE) package --os darwin

# ── CLI only ───────────────────────────────────────────────────────────────────
build-cli:
	go build -o $(BINARY_CLI) $(CMD_CLI)

# ── Windows cross-compile (requires fyne-cross or GOOS override) ──────────────
build-win:
	GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -o $(BINARY_GUI).exe $(CMD_GUI)
	GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -o $(BINARY_CLI).exe $(CMD_CLI)

# ── All ────────────────────────────────────────────────────────────────────────
all: build-unix build-cli

# ── Clean ──────────────────────────────────────────────────────────────────────
clean:
	rm -f $(BINARY_GUI) $(BINARY_GUI).exe
	rm -f $(BINARY_CLI) $(BINARY_CLI).exe
	rm -rf FontLoaderSubRe.app
