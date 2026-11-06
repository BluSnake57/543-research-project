"""Check browser connection URLs without loading models or application dependencies."""

from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import subprocess
import unittest
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


class LandingPage(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.script = ""
        self.in_script = False

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.append(dict(attrs))
        if tag == "script":
            self.in_script = True

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_script = False

    def handle_data(self, data):
        if self.in_script:
            self.script += data


@unittest.skipUnless(shutil.which("node"), "Node.js is needed to execute landing-page JavaScript")
class ContainerProxyTests(unittest.TestCase):
    def test_public_urls_preserve_proxy_origin_and_prefix(self):
        page = LandingPage()
        page.feed((ROOT / "docker/index.html").read_text())
        desktop = next(link for link in page.links if link.get("id") == "desktop-link")
        exports = next(link for link in page.links if "exports/" in link["href"])
        cases = [
            ("http://192.0.2.10:36080/", "/", "36080", "false"),
            ("https://research.example/", "/", "443", "true"),
            ("https://research.example/sentinel/", "/sentinel/", "443", "true"),
            (
                "https://research.example:8443/lab/desktop/?ignored=1",
                "/lab/desktop/",
                "8443",
                "true",
            ),
        ]
        for public_url, prefix, port, encrypt in cases:
            with self.subTest(url=public_url):
                # URL and URLSearchParams are the same APIs used by browsers.
                harness = """
                    const location = new URL(INPUT.url);
                    const link = {href: new URL(INPUT.desktop, location).href};
                    const window = {location};
                    const document = {getElementById: () => link};
                """
                output = subprocess.check_output(
                    [
                        "node",
                        "-e",
                        "const INPUT = "
                        + json.dumps(
                            {
                                "url": public_url,
                                "desktop": desktop["href"],
                                "exports": exports["href"],
                            }
                        )
                        + ";"
                        + harness
                        + page.script
                        + """
                            const result = new URL(link.href);
                            console.log(JSON.stringify({
                                origin: result.origin,
                                pathname: result.pathname,
                                settings: Object.fromEntries(result.searchParams),
                                exports: new URL(INPUT.exports, location).pathname,
                            }));
                        """,
                    ],
                    text=True,
                )
                result = json.loads(output)
                public = urlsplit(public_url)
                self.assertEqual(result["origin"], f"{public.scheme}://{public.netloc}")
                self.assertEqual(result["pathname"], prefix + "vnc.html")
                self.assertEqual(result["exports"], prefix + "exports/")
                settings = result["settings"]
                self.assertEqual(settings["path"], prefix.lstrip("/") + "websockify")
                self.assertEqual(settings["port"], port)
                self.assertEqual(settings["encrypt"], encrypt)
                self.assertEqual(settings["host"], public.hostname)
                self.assertEqual(settings["autoconnect"], "true")
                self.assertNotIn("password", settings)


if __name__ == "__main__":
    unittest.main()
