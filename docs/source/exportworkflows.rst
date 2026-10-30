Export a workflow from the app
==============================

Enable **Settings -> Interface -> Developer mode** and select **Code** on a
workflow page. The download is one ``.py`` file, containing editable settings
and input/output folder definitions.

Uploads are optional: missing inputs use placeholder filenames. Edit these
paths to use files already stored locally, or place the files in the input folder.

Place the listed original files in its input folder, adjust paths as needed,
and run it with the same DNAscape version. Data Manager scripts reproduce
preparation and transforms; Mapping and Simulation scripts reproduce their
numerical outputs. Analysis code reproduces its selected preparation and
output settings, including requested figure exports.


The exported script calls the shared workflow layer, retaining preparation,
resampling, transforms and output choices from the page. A direct ``rmap`` or
``rsim`` call uses the same scientific engine but needs its inputs and
assumptions supplied explicitly; see :doc:`pythonmodules`.

.. code-block:: bash

   python data_manager.py

Use the downloaded filename for Mapping, Simulation or Analysis. The script
defines ``INPUT_DIRECTORY`` and ``OUTPUT_DIRECTORY`` near the top; outputs
are written relative to those folders rather than the terminal's current
directory. Keep the script, source files and DNAscape version with the result.
