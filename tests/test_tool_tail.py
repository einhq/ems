from ems.tools.run import Completed


def failed(stderr: str) -> Completed:
    return Completed(["ffmpeg"], 1, b"", stderr.encode())


def test_a_failure_tail_skips_ffmpegs_closing_summary_for_the_fault():
    done = failed(
        "[mpeg1video @ 0x5a83] Missing picture start code\n"
        "[vost#0:0/libsvtav1 @ 0x5a84] Could not open encoder before EOF\n"
        "[out#0/mp4 @ 0x5a85] video:0KiB audio:43KiB subtitle:0KiB other streams:0KiB global headers:0KiB muxing overhead: unknown\n"
        "frame=    0 fps=0.0 q=85.0 Lsize=       0KiB time=00:00:06.10 bitrate=   0.0kbits/s speed=69.7x elapsed=0:00:00.08    \n"
        "Conversion failed!\n"
    )
    assert done.tail(2) == "[mpeg1video @ 0x5a83] Missing picture start code\n[vost#0:0/libsvtav1 @ 0x5a84] Could not open encoder before EOF"


def test_an_empty_output_keeps_the_line_that_says_so():
    done = failed(
        "[out#0/image2 @ 0x56a2] video:0KiB audio:0KiB subtitle:0KiB other streams:0KiB global headers:0KiB muxing overhead: unknown\n"
        "[out#0/image2 @ 0x56a2] Output file is empty, nothing was encoded(check -ss / -t / -frames parameters if used)\n"
        "frame=    0 fps=0.0 q=0.0 Lsize=N/A time=N/A bitrate=N/A speed=N/A elapsed=0:00:00.00\n"
    )
    assert done.tail() == "[out#0/image2 @ 0x56a2] Output file is empty, nothing was encoded(check -ss / -t / -frames parameters if used)"


def test_other_tools_tails_are_their_last_lines():
    assert failed("one\ntwo\n\nthree\n").tail(2) == "two\nthree"
