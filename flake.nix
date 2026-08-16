{
  description = "NegPy — a tool for processing film negatives";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    pyproject-nix,
    uv2nix,
    pyproject-build-systems,
    ...
  }: let
    inherit (nixpkgs) lib;
    forAllSystems = lib.genAttrs ["x86_64-linux" "aarch64-linux"];

    workspace = uv2nix.lib.workspace.loadWorkspace {workspaceRoot = ./.;};

    overlay = workspace.mkPyprojectOverlay {
      sourcePreference = "wheel";
    };

    editableOverlay = workspace.mkEditablePyprojectOverlay {
      root = "$REPO_ROOT";
    };

    # Packages that need system Qt6/Vulkan/LLVM rather than what autoPatchelf can wire
    # up from a bare PyPI wheel — swap in nixpkgs' own builds instead.
    nixpkgsFallbackOverlay = pkgs: python3Packages: final: prev: let
      hacks = pkgs.callPackage pyproject-nix.build.hacks {};
      fromNixpkgs = attr: prevPkg:
        hacks.nixpkgsPrebuilt {
          from = python3Packages.${attr};
          prev = prevPkg;
        };
    in {
      numpy = fromNixpkgs "numpy" prev.numpy;
      llvmlite = fromNixpkgs "llvmlite" prev.llvmlite;
      numba = fromNixpkgs "numba" prev.numba;
      # nixpkgs' own "opencv-python-headless" is a metapackage that only propagates
      # "opencv4" (where cv2 actually lives); nixpkgsPrebuilt strips propagation, so
      # point straight at opencv4 or the venv ends up without a cv2 module.
      opencv-python-headless = fromNixpkgs "opencv4" prev.opencv-python-headless;
      rawpy = fromNixpkgs "rawpy" prev.rawpy;
      # imagecodecs and tifffile are left off this list deliberately: nixpkgs'
      # imagecodecs has no JPEG-XL codec (breaks JXL export/import round-trips),
      # and nixpkgs' tifffile lags the pinned version enough to break DNG/extrasamples
      # handling. Their PyPI manylinux wheels are self-contained, so plain
      # autoPatchelf (uv2nix's default wheel path) works better than the fallback.
      pillow = fromNixpkgs "pillow" prev.pillow;

      pyqt6-sip = fromNixpkgs "pyqt6-sip" prev.pyqt6-sip;
      pyqt6 = hacks.nixpkgsPrebuilt {
        from = python3Packages.pyqt6;
        # PyQt6-Qt6 (bundled standalone Qt6 libs) has no nixpkgs equivalent and
        # isn't needed — nixpkgs' pyqt6 links the system Qt6 instead.
        prev = prev.pyqt6.overrideAttrs (old: {
          passthru =
            old.passthru
            // {
              dependencies = lib.filterAttrs (name: _: name != "pyqt6-qt6") old.passthru.dependencies;
            };
        });
      };
      pyqt6-charts = hacks.nixpkgsPrebuilt {
        from = python3Packages.pyqt6-charts;
        prev = prev.pyqt6-charts.overrideAttrs (old: {
          passthru =
            old.passthru
            // {
              dependencies = lib.filterAttrs (name: _: name != "pyqt6-charts-qt6") old.passthru.dependencies;
            };
        });
      };

      # nixpkgs' pname is "wgpu-py" but upstream's pyproject.toml declares
      # name = "wgpu", so pythonMetadataCheckPhase can't find dist-info under
      # "wgpu-py" and fails the build. Skip that check.
      wgpu = hacks.nixpkgsPrebuilt {
        from = python3Packages.wgpu-py.overrideAttrs {dontCheckPythonMetadata = true;};
        prev = prev.wgpu;
      };
    };

    # icc/media/crosstalk/gear and VERSION live at the repo root, not inside the
    # negpy/ package tree, so setuptools never picks them up as package data.
    # get_resource_path() (negpy/kernel/system/paths.py) walks 3 levels up from
    # itself and expects to land in the site-packages dir these are copied into.
    # negpy/features/**/shaders/*.wgsl are inside the package but non-.py, so
    # setuptools drops those too — merge the source tree back in after install.
    dataFixupOverlay = python: final: prev: {
      negpy = prev.negpy.overrideAttrs (old: {
        postInstall =
          (old.postInstall or "")
          + ''
            site_packages="$out/${python.sitePackages}"
            cp -r icc media crosstalk gear "$site_packages/"
            cp -r negpy/features "$site_packages/negpy/"
            cp VERSION "$site_packages/"
          '';
      });
    };

    mkPythonSet = pkgs: let
      python = pkgs.python313;
    in
      (pkgs.callPackage pyproject-nix.build.packages {inherit python;}).overrideScope (
        lib.composeManyExtensions [
          pyproject-build-systems.overlays.wheel
          overlay
          (nixpkgsFallbackOverlay pkgs pkgs.python313Packages)
          (dataFixupOverlay python)
        ]
      );

    pythonSets = forAllSystems (system: mkPythonSet nixpkgs.legacyPackages.${system});

    mkNegpy = pkgs: pythonSet: let
      venv = pythonSet.mkVirtualEnv "negpy-env" workspace.deps.default;
    in
      pkgs.stdenv.mkDerivation {
        pname = "negpy";
        version = lib.trim (builtins.readFile ./VERSION);
        dontUnpack = true;
        dontBuild = true;

        nativeBuildInputs = [
          pkgs.makeWrapper
          pkgs.qt6.wrapQtAppsHook
        ];
        buildInputs = [pkgs.qt6.qtbase];
        # We wrap by hand below so we control both the Qt env and LD_LIBRARY_PATH
        # on the same wrapProgram call; wrapQtAppsHook only supplies qtWrapperArgs.
        dontWrapQtApps = true;

        installPhase = ''
          runHook preInstall

          mkdir -p $out/bin
          makeWrapper ${venv}/bin/python $out/bin/negpy \
            --add-flags "-m negpy.desktop.main" \
            "''${qtWrapperArgs[@]}" \
            --prefix LD_LIBRARY_PATH : ${
            lib.makeLibraryPath [
              pkgs.vulkan-loader
              pkgs.libGL
            ]
          }

          install -Dm644 ${./negpy.desktop} $out/share/applications/negpy.desktop
          substituteInPlace $out/share/applications/negpy.desktop \
            --replace-fail "Exec=NegPy" "Exec=$out/bin/negpy" \
            --replace-fail "Icon=icon" "Icon=negpy"
          install -Dm644 ${./media/icons/icon.svg} $out/share/icons/hicolor/scalable/apps/negpy.svg

          runHook postInstall
        '';

        meta = {
          description = "Tool for processing film negatives with film-physics simulation";
          homepage = "https://github.com/marcinz606/NegPy";
          license = lib.licenses.gpl3Only;
          mainProgram = "negpy";
          platforms = lib.platforms.linux;
        };
      };
  in {
    packages = forAllSystems (system: {
      default = mkNegpy nixpkgs.legacyPackages.${system} pythonSets.${system};
    });

    apps = forAllSystems (system: {
      default = {
        type = "app";
        program = "${self.packages.${system}.default}/bin/negpy";
      };
    });

    devShells = forAllSystems (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
        pythonSet = pythonSets.${system}.overrideScope editableOverlay;
        # Matches CI's test job (uv sync --group dev --group plustek --group pieusb):
        # plustek/pieusb are pure Python + bundled libusb, so they build standalone;
        # sane and camera need a system SANE/libgphoto2 install and their tests
        # importorskip away without them, so they're left out here too.
        virtualenv = pythonSet.mkVirtualEnv "negpy-dev-env" {
          negpy = [
            "dev"
            "plustek"
            "pieusb"
          ];
        };
      in {
        default = pkgs.mkShell {
          packages = [
            virtualenv
            pkgs.uv
            pkgs.qt6.qtbase
          ];
          env = {
            UV_NO_SYNC = "1";
            UV_PYTHON = pythonSet.python.interpreter;
            UV_PYTHON_DOWNLOADS = "never";
            # Makes `uv run` (what the Makefile's targets use) treat this venv as
            # the project environment instead of creating its own empty ./.venv.
            # UV_NO_SYNC keeps it from then trying to write into that read-only
            # Nix store path.
            UV_PROJECT_ENVIRONMENT = virtualenv;
            LD_LIBRARY_PATH = lib.makeLibraryPath [
              pkgs.vulkan-loader
              pkgs.libGL
            ];
          };
          shellHook = ''
            unset PYTHONPATH
            export REPO_ROOT=$(git rev-parse --show-toplevel)
          '';
        };
      }
    );
  };
}
