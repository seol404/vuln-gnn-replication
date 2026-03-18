import java.io.PrintWriter

def esc(s: String): String =
  s.replace("\\", "\\\\")
   .replace("\"", "\\\"")
   .replace("\n", " ")
   .replace("\r", " ")

val out = new PrintWriter("ast_graphs.jsonl")

cpg.method.foreach { m =>
  val name = m.name

  val label =
    if (name.contains("bad")) 1
    else if (name.contains("good")) 0
    else -1

  if (label != -1) {
    val ast_nodes = m.ast.l
    val ast_ids = ast_nodes.map(_.id).toSet

    val nodes_json = ast_nodes.map { n =>
      val nid = n.id
      val nlabel = esc(n.label)
      val ncode = esc(Option(n.code).getOrElse(""))
      s"""{"id":$nid,"type":"$nlabel","code":"$ncode"}"""
    }.mkString(",")

    val edges = ast_nodes.flatMap { parent =>
      parent.astChildren.l
        .filter(child => ast_ids.contains(child.id))
        .map(child => s"""{"src":${parent.id},"dst":${child.id}}""")
    }.mkString(",")

    val func_code = esc(Option(m.code).getOrElse(""))

    out.println(
      s"""{"name":"${esc(name)}","label":$label,"code":"$func_code","nodes":[$nodes_json],"edges":[$edges]}"""
    )
  }
}

out.close()
println("Wrote ast_graphs.jsonl")