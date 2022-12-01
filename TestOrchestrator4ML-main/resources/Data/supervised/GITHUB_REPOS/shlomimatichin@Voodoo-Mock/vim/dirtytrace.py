import re
import argparse
import sys

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers( dest = "cmd" )
trace = subparsers.add_parser( "trace", help = "Put a trace line with sorrounding comments, for easy spotting with diff" )
breakpoint = subparsers.add_parser( "breakpoint", help = "Put a breakpoint" )
parser.add_argument( "filename" )
args = parser.parse_args()

inputLines = sys.stdin.readlines()
indent = re.match( r"(\s*)\S", inputLines[ 0 ] ).group( 1 )
input = "".join( inputLines )

if args.cmd == "trace":
    if args.filename.endswith( ".py" ):
        sys.stdout.write( "### DIRTY TRACE\n" +
                            indent + "print 'X'*100\n" +
                            "### DIRTY TRACE END\n" +
                            input )
    elif args.filename.endswith( ".cpp" ) or args.filename.endswith( ".h" ):
        sys.stdout.write( "/// DIRTY TRACE\n" +
                            '''std::cerr << __FILE__ << ':' << __LINE__ << ": XXXX " << std::endl;\n''' +
                            "/// DIRTY TRACE END\n" +
                            input )
    else:
        assert False, "Not implemented for this file type"
elif args.cmd == "breakpoint":
    if args.filename.endswith( ".py" ):
        sys.stdout.write( "### DIRTY BREAKPOINT\n" +
                            indent + "import pdb\n" +
                            indent + "pdb.set_trace()\n" +
                            "### DIRTY BREAKPOINT END\n" +
                            input )
    else:
        assert False, "Not implemented for this file type"
