using Dates

function hfun_bar(vname)
  val = Meta.parse(vname[1])
  return round(sqrt(val), digits=2)
end

function hfun_m1fill(vname)
  var = vname[1]
  return pagevar("index", var)
end

function hfun_blogposts()
  io = IOBuffer()
  posts = sort!(filter!(x -> endswith(x, ".md") && x != "index.md", readdir(joinpath(@__DIR__, "blog"))); rev=true)
  println(posts)
  for post in first.(splitext.(posts))
      url = splitext("/blog/$post/")[1]
      title = pagevar(strip(url, '/'), :title)
      date = Date(post[1:10])
      date ≤ today() && write(io, "\n[$title]($url) $date \n")
  end
  return Franklin.fd2html(String(take!(io)), internal=true)
end

function lx_baz(com, _)
  # keep this first line
  brace_content = Franklin.content(com.braces[1]) # input string
  # do whatever you want here
  return uppercase(brace_content)
end


