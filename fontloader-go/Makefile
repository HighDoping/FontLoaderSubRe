.PHONY: all build-mac build-cli build-win clean

BINARY_GUI    := FontLoaderSubRe
BINARY_CLI    := cli_indexer
CMD_GUI       := ./cmd/fontloader
CMD_CLI       := ./cmd/cli_indexer

# ── macOS ──────────────────────────────────────────────────────────────────────
build-mac: $(BINARY_GUI) $(BINARY_CLI)

$(BINARY_GUI):
	go build -o $(BINARY_GUI) $(CMD_GUI)

$(BINARY_CLI):
	go build -o $(BINARY_CLI) $(CMD_CLI)

# Optional: package as .app bundle using fyne tool
bundle-mac:
	fyne package -os darwin -name FontLoaderSubRe -appID com.highdoping.fontloadersubre -src $(CMD_GUI)

# ── CLI only ───────────────────────────────────────────────────────────────────
build-cli:
	go build -o $(BINARY_CLI) $(CMD_CLI)

# ── Windows cross-compile (requires fyne-cross or GOOS override) ──────────────
build-win:
	GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -o $(BINARY_GUI).exe $(CMD_GUI)
	GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -o $(BINARY_CLI).exe $(CMD_CLI)

# ── Tests & Hygiene ────────────────────────────────────────────────────────────
test:
	go test ./...

vet:
	go vet ./...

tidy:
	go mod tidy

# ── All ────────────────────────────────────────────────────────────────────────
all: build-mac build-cli

# ── Clean ──────────────────────────────────────────────────────────────────────
clean:
	rm -f $(BINARY_GUI) $(BINARY_GUI).exe
	rm -f $(BINARY_CLI) $(BINARY_CLI).exe
	rm -rf FontLoaderSubRe.app
