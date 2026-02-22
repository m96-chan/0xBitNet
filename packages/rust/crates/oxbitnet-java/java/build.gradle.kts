plugins {
    java
    id("com.vanniktech.maven.publish") version "0.30.0"
}

group = "io.github.m96-chan"
version = "0.5.2"

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

mavenPublishing {
    publishToMavenCentral(com.vanniktech.maven.publish.SonatypeHost.CENTRAL_PORTAL)
    signAllPublications()

    pom {
        name.set("OxBitNet")
        description.set("Java bindings for OxBitNet — run BitNet b1.58 ternary LLMs with wgpu")
        url.set("https://github.com/m96-chan/0xBitNet")

        licenses {
            license {
                name.set("MIT License")
                url.set("https://opensource.org/licenses/MIT")
            }
        }

        developers {
            developer {
                id.set("m96-chan")
                name.set("m96-chan")
                url.set("https://github.com/m96-chan")
            }
        }

        scm {
            url.set("https://github.com/m96-chan/0xBitNet")
            connection.set("scm:git:git://github.com/m96-chan/0xBitNet.git")
            developerConnection.set("scm:git:ssh://github.com/m96-chan/0xBitNet.git")
        }
    }
}
