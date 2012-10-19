IPython started as a better interactive Python interpreter in 2001,
but over the last decade it has grown into a rich and powerful set of
interlocking tools aimed at maximizing developer productivity with
Python while using the language interactively.

Today, IPython consists of a kernel that executes the user code and
controls the user's namespace, and a collection of tools to control
this kernel either in-process or out-of-process thanks to a
well-specified communications protocol implemented over ZeroMQ.  The
kernel can do much more than execute user code, including
introspection of objects in the user's namespace, detailed error
reporting with rich tracebacks, history logging of inputs and outputs
with an SQLite backend, a user-extensible system of commands for
interactive control that don't collide with user variables, and much
more.  

Our communications architecture allows these same features to be
accessed via a variety of clients, each providing unique functionality
tuned to a specific use case.  We expose a number of directly usable
applications:

- An interactive, terminal-based shell with many capabilities far
beyond the default Python interactive interpreter (this is the default
application opened by the ipython command that most users are familiar
with).

- A Qt console that provides the look and feel of a terminal, but adds
support for inline figures, graphical calltips, a persistent session
that can survive crashes (even segfaults) of the kernel process, and
more.  A user-based review of some of these features can be found
[here](http://stronginference.com/weblog/2011/7/15/innovations-in-ipython.html).

- A web-based notebook that can execute code and also contain rich
text and figures, mathematical equations and arbitrary HTML.  This
notebook controls the same kernel as the other two applications, but
instead of offering a linear, terminal-like workflow, it presents a
document-like view with cells where code is executed but that can be
edited in-place, reordered, mixed with explanatory text and figures,
etc.  This model is a kind of literate programming environment popular
in scientific computing and pioneered by the Mathematica system, that
allows for the creation of rich documents that combine computational
experimentation and results with other explanatory elements.  A
detailed review of this system can be found
[here](http://lighthouseinthesky.blogspot.com/2011/09/review-ipython-notebooks.html).

- A high-performance, low-latency system for parallel computing that
supports the control of a cluster of IPython engines communicating
over ZeroMQ, with optimizations that minimize unnecessary copying of
large objects (especially numpy arrays).  These engines can be
controlled interactively while developing and doing exploratory work,
or can run in batch mode either on a local machine or in a large
cluster/supercomputing environment via a batch scheduler.

In this hands-on, in-depth tutorial, we will briefly describe
IPython's architecture  and will then show how to use and configure
each of the above components.  We will also discuss how to use the
underlying IPython libraries in your own application to provide
interactive control.  

An outline of the tutorial follows:

- Introductory description of the project and architecture.
- IPython basics: the magic command system, shell aliases, full shell
access, the history system, variable caching, object introspection
tools.
- Development workflow: combining the interpreter session with python
files via the %run command.
- Effective use of IPython at the command-line for typical development
tasks: timing, profiling, debugging.
- Embedding IPython in terminal applications.
- The IPython Qt console: unique features beyond the terminal.
- Embedding an IPython kernel in a GUI app to expose network-based
interactive control.
- Configuring IPython: the profile and configuration system for
multiple applications.
- The IPython notebook: interactive usage of the application, the
IPython display protocol, defining custom display methods for your own
objects, generating HTML and PDF output.
- Parallelism with IPython: basic architecture, interactive control of
a cluster, standalone execution of applications, integration with MPI,
blocking and asynchronous parallelism, execution in batch-controlled
environments, IPython engines in the cloud (illustrated with Amazon
EC2 instances).
- A short listing of other features not covered in this tutorial, as
guidance for users to later learn about on their own.

For full details about IPython including documentation, previous
presentations and videos of talks, please see the [project
website](http://ipython.org).
