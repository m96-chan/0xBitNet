// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "OxBitNet",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
    ],
    products: [
        .library(name: "OxBitNet", targets: ["OxBitNet"]),
    ],
    targets: [
        .target(
            name: "COxBitNet",
            path: "Sources/COxBitNet"
        ),
        .target(
            name: "OxBitNet",
            dependencies: ["COxBitNet"],
            path: "Sources/OxBitNet"
        ),
        .executableTarget(
            name: "Chat",
            dependencies: ["OxBitNet"],
            path: "Examples/Chat"
        ),
    ]
)
