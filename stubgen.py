import sys
from typing import IO


def process(fin: IO[str], fout: IO[str]) -> None:
    gap = 0
    max_len = 79
    prev_is_symbol = False
    add_indent = " " * 4
    buff = ""
    pretext_buff = ""

    def write_pretext(text: str) -> None:
        nonlocal pretext_buff

        pretext_buff = f"{pretext_buff}{text}"

    def write_line(text: str) -> None:
        nonlocal buff
        nonlocal gap
        nonlocal pretext_buff

        if not text.strip():
            if gap < 2:
                buff = f"{buff}{text}"
                gap += 1
            return
        if buff:
            fout.write(buff)
            buff = ""
        if pretext_buff:
            fout.write(pretext_buff)
            pretext_buff = ""
        fout.write(text)
        fout.flush()
        gap = 0

    start = True
    for line in fin:
        if not line.strip():
            write_line(line)
            continue

        if start and line.startswith("# pylint: disable"):
            continue
        start = False

        if line.lstrip().startswith("@"):
            write_pretext(line)
            continue

        is_indent = line.startswith(" ")
        is_def = line.lstrip().startswith("def ")
        is_class_def = is_def and not line.startswith("def ")
        is_class = line.startswith("class ")
        is_dddot = line.rstrip().endswith("...")
        is_need_gap = not (is_class or is_def) and not prev_is_symbol
        prev_is_symbol = not is_class and not is_def and not is_class_def
        if is_indent:
            cur_indent = line[:line.index(line.lstrip()[0])]
        else:
            cur_indent = ""
        if not is_indent:
            if is_class or is_def or is_need_gap:
                write_line("\n")
                write_line("\n")

        if is_class and is_dddot:
            class_def = line.rstrip().rstrip(". ")
            if len(class_def) > max_len:
                raise ValueError(f"class def longer than line! {class_def}")
            write_line(f"{class_def}\n{add_indent}...\n")
            continue

        outline = line.rstrip()
        if len(outline) <= max_len:
            write_line(line)
            continue

        if is_class_def and gap < 1:
            write_line("\n")

        first_paren = outline.find("(") + 1
        in_paren = False
        if first_paren > 0:
            write_line(f"{outline[:first_paren]}\n")
            outline = f"{cur_indent}{add_indent}{outline[first_paren:]}"
            in_paren = True
        while len(outline) > max_len or not outline.strip():
            bslash = ""
            end_ix = max_len - 2
            right_cutoff = outline.rfind(",", 0, end_ix) + 1
            if right_cutoff <= 0:
                right_cutoff = outline.rfind(":", 0, end_ix) + 1
                if right_cutoff <= 0:
                    right_cutoff = outline.rfind(" ", 0, end_ix)
                    if right_cutoff < 0:
                        raise ValueError(
                            "could not find any way to shorten line: "
                            f"{outline}")
                    bslash = " \\"
            if outline.find(")", 0, right_cutoff) >= 0:
                in_paren = False
            if in_paren:
                extra_indent = add_indent
            else:
                extra_indent = f"{add_indent}{add_indent}"
            write_line(f"{outline[:right_cutoff].rstrip()}{bslash}\n")
            outline = \
                f"{cur_indent}{extra_indent}{outline[right_cutoff:].lstrip()}"
        if outline.strip():
            write_line(f"{outline}\n")

        if is_class_def:
            write_line("\n")

    buff = buff.strip()
    if buff:
        fout.write(f"{buff}\n")
        fout.flush()


def run(fname_in: str, fname_out: str) -> None:
    with open(fname_in, "r", encoding="utf-8") as fin:
        with open(fname_out, "a", encoding="utf-8") as fout:
            process(fin, fout)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("must be called from stubgen.sh")
    run(sys.argv[1], sys.argv[2])
