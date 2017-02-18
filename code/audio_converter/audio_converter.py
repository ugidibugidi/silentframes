#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Helper functions to decode audio files.

Authors: Jan Schlüter (decode_to_disk, decode_to_memory), Reinhard Sonnleitner (audio_converter)
"""

import os
import sys
import tempfile
import subprocess

def decode_to_disk(soundfile, fmt='f32le', sample_rate=None, skip=None, maxlen=None, outfile=None, tmpdir=None, tmpsuffix=None):
	"""
	Decodes the given audio file, downmixes it to mono and
	writes it to another file as a sequence of samples.
	Returns the file name of the output file.
	@param soundfile: The sound file to decode
	@param fmt: The format of samples:
		'f32le' for float32, little-endian.
		's16le' for signed 16-bit int, little-endian.
	@param sample_rate: The sample rate to resample to.
	@param skip: Number of seconds to skip at beginning of file.
	@param maxlen: Maximum number of seconds to decode.
	@param outfile: The file to decode the sound file to. If not
		given, a temporary file will be created.
	@param tmpdir: The directory to create the temporary file in
		if no outfile was given.
	@param tmpsuffix: The file extension for the temporary file if
		no outfile was given. Example: ".pcm" (include the dot!)
	@return The output file name.
	"""
	# create temp file if no outfile is given
	if outfile is None:
		# Looks stupid, but is recommended over tempfile.mktemp()
		f = tempfile.NamedTemporaryFile(delete=False, dir=tmpdir, suffix=tmpsuffix)
		f.close()
		outfile = f.name
		delete_on_fail = True
	else:
		delete_on_fail = False
	# call ffmpeg (throws exception on error)
	try:
		call = _assemble_ffmpeg_call(soundfile, outfile, fmt, sample_rate, skip, maxlen)
		subprocess.check_call(call)
	except Exception:
		if delete_on_fail:
			os.unlink(outfile)
		raise
	return outfile

def decode_to_memory(soundfile, fmt='f32le', sample_rate=None, skip=None, maxlen=None):
	"""
	Decodes the given audio file, downmixes it to mono and
	returns it as a binary string of a sequence of samples.
	@param soundfile: The sound file to decode
	@param fmt: The format of samples:
		'f32le' for float32, little-endian.
		's16le' for signed 16-bit int, little-endian.
	@param sample_rate: The sample rate to resample to.
	@param skip: Number of seconds to skip at beginning of file.
	@param maxlen: Maximum number of seconds to decode.
	@return A binary string of samples or an MP3 file.
	"""
	call = _assemble_ffmpeg_call(soundfile, "pipe:1", fmt, sample_rate, skip, maxlen)
	if hasattr(subprocess, 'check_output'):
		# call ffmpeg (throws exception on error)
		signal = subprocess.check_output(call)
	else:
		# this is an old version of Python, do subprocess.check_output manually
		proc = subprocess.Popen(call, stdout=subprocess.PIPE, bufsize=-1)
		signal, _ = proc.communicate()
		if proc.returncode != 0:
			raise subprocess.CalledProcessError(proc.returncode, call)
	return signal

def _assemble_ffmpeg_call(infile, output, fmt='f32le', sample_rate=None, skip=None, maxlen=None):
	"""
	Internal function. Creates a sequence of strings indicating the application
	(ffmpeg) to be called as well as the parameters necessary to decode the
	given input file to the given format, at the given offset and for the given
	length to the given output.
	"""
	if isinstance(infile, unicode):
		infile = infile.encode(sys.getfilesystemencoding())
	else:
		infile = str(infile)
	call = ["ffmpeg", "-v", "-10", "-y", "-i", infile, "-f", str(fmt), "-ac", "1"]
	if sample_rate is not None:
		call.extend(["-ar", str(sample_rate)])
	if skip is not None:
		call.extend(["-ss", str(float(skip))])
	if maxlen is not None:
		call.extend(["-t", str(float(maxlen))])
	call.append(output)
	return call




class audio_converter:
    def __init__(self, filename, tmpdir, startpos=None, duration=None):
        self.infile = filename
        self.startpos=startpos
        self.duration=duration
        self.tmpdir=tmpdir
    
    def convert_ffmpeg(self):
        "calls script to convert mp3 to wav"
        program = "../util/convert_ffmpeg.sh" # perhaps the absolute path is needed
        proc_convert_params = [self.infile, self.tmpdir]
        if self.startpos is not None and self.duration is not None:
            print self.startpos
            print self.duration
            proc_convert_params.append(`self.startpos`)
            proc_convert_params.append(`self.duration`)
        nulfp = open(os.devnull, "w")
        proc = subprocess.Popen([program]+proc_convert_params,  shell=False, stdout=subprocess.PIPE, stderr=nulfp.fileno())
        msg = str(proc.communicate())
        pcmfile = self.infile.rsplit('.',1)[0]+".pcm"
        
        #using ffmpeg f32le samples, we use the suffix ".pcm"
        return msg, pcmfile

    def convert(self):
        msg, pcmfilename=self.convert_ffmpeg()
        return msg, pcmfilename

