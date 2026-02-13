#!/usr/bin/env python3
"""
Compare two files line-by-line:
1. Forward: First line where content differs (top to bottom).
2. Reverse: First line where content differs when comparing from the end (bottom to top).

Note: Files may have different lengths.
"""

import sys

def read_lines(filepath):
    """Read file and return list of lines with trailing newlines stripped."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\n\r') for line in f]
        return lines
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

def find_first_forward_divergence(lines1, lines2):
    """Returns (line_num_file1, line_num_file2) of first divergence or None if identical."""
    len1, len2 = len(lines1), len(lines2)
    
    # Compare up to min(len1, len2)
    for i in range(min(len1, len2)):
        if lines1[i] != lines2[i]:
            return (i + 1, i + 1)  # same line index, but both are 1-based

    # If we reach here: all overlapping lines match.
    # Divergence = where shorter file ends (first extra line in longer file)
    if len1 != len2:
        if len1 > len2:
            return (len2 + 1, None)  # file1 has an extra line at position len2+1
        else:
            return (None, len1 + 1)  # file2 has an extra line at position len1+1

    return (-1, -1)  # identical

def find_first_reverse_divergence(lines1, lines2):
    """
    Compares from the end: 
      compare last lines → second-to-last → ... until mismatch
    Returns (line_num_file1, line_num_file2)
    If one file is shorter, we stop comparing when its lines run out.
    The *first* extra line (from bottom) in the longer file is considered a divergence.
    """
    len1, len2 = len(lines1), len(lines2)

    # Compare from end: index -i means last, second-to-last, etc.
    max_reverse_compare = min(len1, len2)
    
    for i in range(1, max_reverse_compare + 1):
        if lines1[-i] != lines2[-i]:
            return (len1 - i + 1, len2 - i + 1)

    # If all common lines match but lengths differ:
    if len1 != len2:
        if len1 > len2:
            # First extra line in file1 is at position: len2+1 (forward)
            return (len2 + 1, None)
        else:
            return (None, len1 + 1)

    return (-1, -1)  # identical

def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_lines.py <file1> <file2>", file=sys.stderr)
        sys.exit(1)

    f1_path, f2_path = sys.argv[1], sys.argv[2]

    lines1 = read_lines(f1_path)
    lines2 = read_lines(f2_path)

    # --- Forward comparison ---
    fwd = find_first_forward_divergence(lines1, lines2)
    print("Forward (top → bottom) divergence:")
    if fwd == (-1, -1):
        print("  Files are identical.")
    elif None in fwd:
        f1_line, f2_line = fwd
        if f1_line is not None:
            print(f"  File1 has extra content starting at line {f1_line} (File2 ends earlier)")
        else:  # f2_line is not None
            print(f"  File2 has extra content starting at line {f2_line} (File1 ends earlier)")
    else:
        l1, l2 = fwd
        print(f"  Diverges at file1 line {l1}, file2 line {l2}")

    # --- Reverse comparison ---
    rev = find_first_reverse_divergence(lines1, lines2)
    print("\nReverse (bottom → top) divergence:")
    if rev == (-1, -1):
        print("  Files are identical.")
    elif None in rev:
        f1_line, f2_line = rev
        if f1_line is not None:
            print(f"  File1 diverges first from bottom at line {f1_line} (File2 too short)")
        else:  # f2_line is not None
            print(f"  File2 diverges first from bottom at line {f2_line} (File1 too short)")
    else:
        l1, l2 = rev
        print(f"  Diverges at file1 line {l1}, file2 line {l2}")

if __name__ == "__main__":
    main()
