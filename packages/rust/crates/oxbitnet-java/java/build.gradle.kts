plugins {
    java
    `maven-publish`
    signing
}

group = "io.github.m96-chan"
version = "0.3.1"

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
    withSourcesJar()
    withJavadocJar()
}

publishing {
    publications {
        create<MavenPublication>("maven") {
            from(components["java"])

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
    }

    repositories {
        maven {
            name = "sonatype"
            val releasesUrl = "https://s01.oss.sonatype.org/service/local/staging/deploy/maven2/"
            val snapshotsUrl = "https://s01.oss.sonatype.org/content/repositories/snapshots/"
            url = uri(if (version.toString().endsWith("SNAPSHOT")) snapshotsUrl else releasesUrl)

            credentials {
                username = findProperty("mavenCentralUsername") as String?
                    ?: System.getenv("MAVEN_CENTRAL_USERNAME")
                password = findProperty("mavenCentralPassword") as String?
                    ?: System.getenv("MAVEN_CENTRAL_PASSWORD")
            }
        }
    }
}

signing {
    val signingKey = System.getenv("GPG_PRIVATE_KEY")
    val signingPassword = System.getenv("GPG_PASSPHRASE")
    if (signingKey != null && signingPassword != null) {
        useInMemoryPgpKeys(signingKey, signingPassword)
    }
    setRequired { gradle.taskGraph.hasTask("publish") }
    sign(publishing.publications["maven"])
}
