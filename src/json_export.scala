import java.io.PrintWriter

val out = new PrintWriter("juliet_dataset.jsonl")

cpg.method.foreach { m =>
  val name = m.name
  val label =
    if (name.contains("bad")) 1
    else if (name.contains("good")) 0
    else -1

  if (label != -1) {
    val safeCode = m.code
      .replace("\\", "\\\\")
      .replace("\"", "\\\"")
      .replace("\n", " ")

    out.println(s"""{"name":"$name","label":$label,"code":"$safeCode"}""")
  }
}

out.close()
println("Wrote juliet_dataset.jsonl")