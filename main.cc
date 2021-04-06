#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

using namespace dealii;

template <int dim, int spacedim>
std::vector<std::shared_ptr<Triangulation<dim, spacedim> const>>
create_geometric_coarsening_sequence(
  Triangulation<dim, spacedim> const &fine_triangulation_in)
{
  std::vector<std::shared_ptr<Triangulation<dim, spacedim> const>>
    coarse_grid_triangulations(fine_triangulation_in.n_global_levels());

  coarse_grid_triangulations.back().reset(&fine_triangulation_in, [](auto *) {
    // empty deleter, since fine_triangulation_in is an external field
    // and its destructor is called somewhere else
  });

  // for a single level nothing has to be done
  if (fine_triangulation_in.n_global_levels() > 1)
    {
      auto const fine_triangulation = dynamic_cast<
        parallel::distributed::Triangulation<dim, spacedim> const *>(
        &fine_triangulation_in);

      Assert(fine_triangulation, ExcNotImplemented());

      // clone distributed triangulation for coarsening
      parallel::distributed::Triangulation<dim, spacedim> temp_tria(
        fine_triangulation->get_communicator(),
        fine_triangulation->get_mesh_smoothing());

      temp_tria.copy_triangulation(*fine_triangulation);

      // create coarse meshes
      for (unsigned int l = (fine_triangulation->n_global_levels() - 1); l > 0;
           --l)
        {
          // coarsen mesh
          temp_tria.coarsen_global();

          // create empty (fully distributed) triangulation
          auto new_tria = std::make_shared<
            parallel::fullydistributed::Triangulation<dim, spacedim>>(
            fine_triangulation->get_communicator());

          for (auto const i : fine_triangulation->get_manifold_ids())
            if (i != numbers::flat_manifold_id)
              new_tria->set_manifold(i, fine_triangulation->get_manifold(i));

          if (l - 1 == 0)
            {
              unsigned int const n_partitions = std::min<unsigned int>(
                5,
                Utilities::MPI::n_mpi_processes(
                  fine_triangulation->get_communicator()));
              unsigned int const group_size = [&]() {
                auto comm = temp_tria.get_communicator();

                int rank;
                MPI_Comm_rank(comm, &rank);

                MPI_Comm comm_shared;
                MPI_Comm_split_type(comm,
                                    MPI_COMM_TYPE_SHARED,
                                    rank,
                                    MPI_INFO_NULL,
                                    &comm_shared);

                int size_shared;
                MPI_Comm_size(comm_shared, &size_shared);

                // determine maximum, since some shared memory communicators
                // might not be filed completely
                int size_shared_max;
                MPI_Allreduce(
                  &size_shared, &size_shared_max, 1, MPI_INT, MPI_MAX, comm);

                MPI_Comm_free(&comm_shared);

                return size_shared_max;
              }();

              // extract relevant information from distributed triangulation
              auto const construction_data =
                TriangulationDescription::Utilities::
                  create_description_from_triangulation_in_groups<dim, dim>(
                    [&](auto &tria) {
                      auto const coarse_mesh_description =
                        GridTools::get_coarse_mesh_description(temp_tria);
                      tria.create_triangulation(
                        std::get<0>(coarse_mesh_description),
                        std::get<1>(coarse_mesh_description),
                        std::get<2>(coarse_mesh_description));
                    },
                    [&](auto &tria, auto const &, const auto) {
                      GridTools::partition_triangulation_zorder(n_partitions,
                                                                tria);
                      // GridTools::partition_triangulation(n_partitions, tria);
                    },
                    temp_tria.get_communicator(),
                    group_size);

              // actually create triangulation
              new_tria->create_triangulation(construction_data);
            }
          else
            {
              // extract relevant information from distributed triangulation
              auto const construction_data = TriangulationDescription::
                Utilities::create_description_from_triangulation(
                  temp_tria, fine_triangulation->get_communicator());

              // actually create triangulation
              new_tria->create_triangulation(construction_data);
            }

          // save mesh
          coarse_grid_triangulations[l - 1] = new_tria;
        }
    }

  return coarse_grid_triangulations;
}



int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  const int                                 dim = 2;
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);

  GridGenerator::subdivided_hyper_cube(tria, 10, 0, 1, true);
  tria.refine_global(1);

  const auto v = create_geometric_coarsening_sequence(tria);

  {
    GridOutFlags::Vtk flags;
    flags.output_cells         = false;
    flags.output_faces         = true;
    flags.output_edges         = true;
    flags.output_only_relevant = false;

    GridOut       grid_out_2;
    std::ofstream out(
      "boundary-" +
      std::to_string(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) +
      ".vtk");
    grid_out_2.set_flags(flags);
    grid_out_2.write_vtk(*v[0], out);
  }

  {
    GridOutFlags::Vtk flags;
    flags.output_cells         = true;
    flags.output_faces         = false;
    flags.output_edges         = false;
    flags.output_only_relevant = false;

    GridOut       grid_out_2;
    std::ofstream out(
      "grid-" +
      std::to_string(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) +
      ".vtk");
    grid_out_2.set_flags(flags);
    grid_out_2.write_vtk(*v[0], out);
  }
}
